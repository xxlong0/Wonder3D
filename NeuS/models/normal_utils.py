import numpy as np

def camNormal2worldNormal(rot_c2w, camNormal):
    H,W,_ = camNormal.shape
    normal_img = np.matmul(rot_c2w[None, :, :], camNormal.reshape(-1,3)[:, :, None]).reshape([H, W, 3])

    return normal_img

def worldNormal2camNormal(rot_w2c, normal_map_world):
    H,W,_ = normal_map_world.shape
    # normal_img = np.matmul(rot_w2c[None, :, :], worldNormal.reshape(-1,3)[:, :, None]).reshape([H, W, 3])

    # faster version
    # Reshape the normal map into a 2D array where each row represents a normal vector
    normal_map_flat = normal_map_world.reshape(-1, 3)

    # Transform the normal vectors using the transformation matrix
    normal_map_camera_flat = np.dot(normal_map_flat, rot_w2c.T)

    # Reshape the transformed normal map back to its original shape
    normal_map_camera = normal_map_camera_flat.reshape(normal_map_world.shape)

    return normal_map_camera

def trans_normal(normal, RT_w2c, RT_w2c_target):

    # normal_world = camNormal2worldNormal(np.linalg.inv(RT_w2c[:3,:3]), normal)
    # normal_target_cam = worldNormal2camNormal(RT_w2c_target[:3,:3], normal_world)

    relative_RT = np.matmul(RT_w2c_target[:3,:3], np.linalg.inv(RT_w2c[:3,:3]))
    normal_target_cam = worldNormal2camNormal(relative_RT[:3,:3], normal)

    return normal_target_cam

def img2normal(img):
    return (img/255.)*2-1

def normal2img(normal):
    return np.uint8((normal*0.5+0.5)*255)

def norm_normalize(normal, dim=-1):

    normal = normal/(np.linalg.norm(normal, axis=dim, keepdims=True)+1e-6)

    return normal