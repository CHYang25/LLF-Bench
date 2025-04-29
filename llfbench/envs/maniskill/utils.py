import torch

def quaternion_angle(q_a, q_b):
    if not isinstance(q_a, torch.Tensor):
        q_a = torch.tensor(q_a, dtype=torch.float32)
        q_b = torch.tensor(q_b, dtype=torch.float32)

    dot_product = torch.sum(q_a * q_b)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)  # For numerical stability
    angle = 2 * torch.acos(torch.abs(dot_product))
    return angle 

def quaternion_inverse(q):
    w, x, y, z = q
    return torch.tensor([w, -x, -y, -z], dtype=q.dtype)

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.tensor([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=q1.dtype)

def quaternion_to_euler(q):
    # Assuming quaternion is normalized
    w, x, y, z = q

    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if torch.abs(sinp) >= 1:
        pitch = torch.sign(sinp) * (torch.pi / 2)  # use 90 degrees if out of range
    else:
        pitch = torch.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw])  # (roll, pitch, yaw)

def quaternion_rotation_difference(q1, q2):
    q1 = torch.tensor(q1, dtype=torch.float32)
    q2 = torch.tensor(q2, dtype=torch.float32)
    
    # Ensure normalized
    q1 = q1 / torch.norm(q1)
    q2 = q2 / torch.norm(q2)
    
    q1_inv = quaternion_inverse(q1)
    q_rel = quaternion_multiply(q1_inv, q2)
    
    return quaternion_to_euler(q_rel)