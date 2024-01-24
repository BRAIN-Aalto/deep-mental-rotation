from typing import Optional, List

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import wandb

from misc.dataloaders import (
    SceneRenderDataset,
    create_batch_from_data_list,

)
from models.neural_renderer import NeuralRenderer


def get_dataset_psnr(
    device: torch.device,
    model: NeuralRenderer,
    dataset: SceneRenderDataset,
    source_img_idx_shift: int = 64,
    batch_size: int = 10,
    max_num_scenes: Optional[int] = None
) -> List[float]:
    """Returns PSNR for each scene in a dataset by comparing the view predicted
    by a model and the ground truth view.

    Args:
        device (torch.device):
            Device to perform PSNR calculation on.

        model (models.neural_renderer.NeuralRenderer):
            Model to evaluate.
        dataset (misc.dataloaders.SceneRenderDataset):
            Dataset to evaluate model performance on.
            Should be one of "chairs-test" or "cars-test".

        source_img_idx_shift (int):
            Index of source image for each scene.
            For example if 00064.png is the source view, then
            source_img_idx_shift = 64.

        batch_size (int):
            Batch size to use when generating predictions. This
            should be a divisor of the number of images per scene.

        max_num_scenes (None or int):
        Optionally limit the maximum number of scenes to calculate PSNR for.

    Notes:
        This function should be used with the ShapeNet chairs and cars *test*
        sets.
    """
    def calculate_batch_psnr(batch_data: List) -> float:
        """
        """
        # Create batch for target data
        img_target, azimuth_target, elevation_target = create_batch_from_data_list(batch_data)
        
        img_target = img_target.to(device)
        azimuth_target = azimuth_target.to(device)
        elevation_target = elevation_target.to(device)

        # Rotate scene and render image
        rotated = model.rotate_source_to_target(
            scenes,
            azimuth_source,
            elevation_source,
            azimuth_target,
            elevation_target
        )
        img_predicted = model.render(rotated).detach()

        log_batch_prediction(img_target, img_predicted)

        return psnr(img_predicted, img_target)


    def log_batch_prediction(
        target_images: torch.Tensor,
        predicted_images: torch.Tensor
    ) -> None:
        """
        """
        comparison = []

        for target, prediction in zip(target_images, predicted_images):
            # calculate a PSNR score for each predicted image
            psnr_paired = psnr(
                torch.unsqueeze(target, 0),
                torch.unsqueeze(prediction, 0)
            )
            # make an image grid with the source view, target view,
            # and predicted view
            img_grid = make_grid(
                [img_source[0], target, prediction],
                nrow=3,
                padding=4
            )
        
            comparison.append(
                wandb.Image(
                    img_grid.cpu().numpy().transpose(1, 2, 0),
                    caption=f"left - source, \
                            middle - target, \
                            right - predicted\n \
                            PSNR = {psnr_paired:.3f}"
                )
            )

        wandb.log({
            f"shape {dataset[source_img_idx]['scene_name'].split('_')[-1]}": comparison
        })


    num_imgs_per_scene = dataset.num_imgs_per_scene
    # Set number of scenes to calculate
    num_scenes = dataset.num_scenes

    if max_num_scenes is not None:
        num_scenes = min(max_num_scenes, num_scenes)
    # Calculate number of batches per scene
    assert (num_imgs_per_scene - 1) % batch_size == 0, \
        f"Batch size ({batch_size}) must divide number of images \
        per scene ({num_imgs_per_scene})."
    
    # Comparison are made against all images except the source image (and
    # therefore subtract 1 from total number of images) 
    batches_per_scene = (num_imgs_per_scene - 1) // batch_size

    # Initialize psnr values
    psnrs = []

    for i in range(num_scenes):
        # Extract source view
        source_img_idx = i * num_imgs_per_scene + source_img_idx_shift

        img_source = dataset[source_img_idx]["img"].unsqueeze(0) \
                                                   .repeat(batch_size, 1, 1, 1) \
                                                   .to(device)
        render_params = dataset[source_img_idx]["render_params"]
        azimuth_source = torch.Tensor([render_params["azimuth"]]).repeat(batch_size) \
                                                                 .to(device)
        elevation_source = torch.Tensor([render_params["elevation"]]).repeat(batch_size) \
                                                                     .to(device)
        # Infer source scene
        scenes = model.inverse_render(img_source)

        # Iterate over all other views of scene
        data_list = []
        num_points_in_batch = 0
        
        scene_psnr = 0.

        for j in range(num_imgs_per_scene):
            if j == source_img_idx_shift:
                continue  # Do not compare against same image

            # Add new image to list of images we want to compare to
            data_list.append(dataset[i * num_imgs_per_scene + j])
            num_points_in_batch += 1

            # If we have filled up a batch, make psnr calculation
            if num_points_in_batch == batch_size:

                scene_psnr += calculate_batch_psnr(data_list)
                
                data_list = []
                num_points_in_batch = 0

        psnrs.append(scene_psnr / batches_per_scene)

        print(
            f"{i + 1:02d}/{num_scenes}: \
            Current - {psnrs[-1]:.5f}, \
            Mean - {torch.mean(torch.Tensor(psnrs)):.5f}"
        )

    return psnrs


def psnr(
    target: torch.Tensor,
    prediction: torch.Tensor
) -> float:
    """Returns PSNR between a batch of predictions and a batch of targets.

    Args:
        target (torch.Tensor):
            Shape (batch_size, channels, height, width).

        prediction (torch.Tensor):
            Shape (batch_size, channels, height, width).

    """
    batch_size = prediction.shape[0]
    mse_per_pixel = F.mse_loss(prediction, target, reduction='none')
    mse_per_img = mse_per_pixel.view(batch_size, -1).mean(dim=1)
    psnr = 10 * torch.log10(1 / mse_per_img)

    return torch.mean(psnr).item()
