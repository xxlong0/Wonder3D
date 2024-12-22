from tqdm import tqdm
from MVMeshRecon.MeshRecon.opt import MeshOptimizer
from remeshing.core.remesh import calc_vertex_normals
from utils.loss_utils import NormalLoss
import torchvision.utils as vutils

def save_tensor_mask(mask, filename):
    # Squeeze to remove any extra dimensions
    mask = mask.squeeze(0)

    # Ensure the mask is in a 0-1 range and in the right shape (CxHxW)
    vutils.save_image(mask, filename)

def do_optimize(vertices, faces, ref_images, renderer, weights, remeshing_steps, edge_len_lims=(0.01, 0.1), decay=0.999):
    # optimizer initialization
    opt = MeshOptimizer(vertices, faces, local_edgelen=False, edge_len_lims=edge_len_lims, gain=0.1)
    vertices = opt.vertices

    # normal optimization step
    loss_func = NormalLoss(mask_loss_weights = 1.)
    for i in tqdm(range(remeshing_steps)):
        opt.zero_grad()
        opt._lr *= decay

        normals = calc_vertex_normals(vertices, faces)
        render_normal = renderer.render_normal(vertices, normals, faces)

        loss_expand = 0.5 * ((vertices + normals).detach() - vertices).pow(2).mean()

        # Extract mask and ground truth mask
        mask = render_normal[..., [3]]
        gtmask = ref_images[..., [3]]

        # Compute loss with the mask
        loss = loss_func(render_normal, ref_images, weights=weights, mask=mask, gtmask=gtmask)
        loss_expansion_weight = 0.1
        loss = loss + loss_expansion_weight * loss_expand

        loss.backward()
        opt.step()
        vertices, faces = opt.remesh(poisson=False)

    return vertices.detach(), faces.detach()