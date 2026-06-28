"""
3D 旋转表示法之间的转换工具（从 transforms3d 库精简抄来）。

机器人姿态（夹爪朝向）在数学上可以用多种等价方式描述：
  - 四元数 quaternion  (w, x, y, z)：仿真器内部常用，无万向锁
  - 旋转矩阵 matrix    3x3
  - 欧拉角 euler       (roll, pitch, yaw)：人最直观
  - 轴角 axis-angle    绕某根轴转 theta 角：仿真器 step 接口要的格式

本文件就是这些表示之间的互转函数。在 YuePi0 部署里用到三条链：
  输入端 preprocess：四元数 --quat2mat--> 矩阵 --mat2euler--> 欧拉角（喂给模型）
  输出端 postprocess：模型吐的欧拉角 --euler2axangle--> 轴角（喂给仿真器）

axes 参数（如 "sxyz"）指定欧拉角的旋转轴顺序和约定，默认 "sxyz"，全程用这个即可。
下面那些 _AXES2TUPLE / parity / repetition / frame 是支持 24 种欧拉约定的通用机制，
看不懂可以跳过——我们只用默认 sxyz，知道函数“输入什么、输出什么”就够了。
"""

import math

import numpy as np

_FLOAT_EPS = np.finfo(np.float64).eps  # float64 的最小精度，用来判断“是否接近 0”

# 欧拉角轴序列的下标循环表：用于在 x/y/z 三轴间按顺序取“下一个轴”
_NEXT_AXIS = [1, 2, 0, 1]

# 把欧拉约定字符串（如 "sxyz"）映射成内部 4 元组：
#   (firstaxis 首轴, parity 奇偶, repetition 是否重复轴, frame 静/动坐标系)
# 24 种是旋转约定的全部组合，我们只用 "sxyz"
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}

# 反向映射：内部 4 元组 -> 约定字符串
_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

# 判断浮点数是否接近 0 的阈值（用于避免除零 / 退化情况）
_EPS4 = np.finfo(float).eps * 4.0


def mat2euler(mat, axes="sxyz"):
    """旋转矩阵 -> 欧拉角。

    输入：
        mat   3x3（或 4x4 取左上 3x3）旋转矩阵
        axes  欧拉约定，默认 sxyz
    输出：
        (ax, ay, az) 三个欧拉角（弧度）

    用途：部署 preprocess 里，把夹爪朝向矩阵转成模型训练时用的欧拉角。
    注意：同一个矩阵可能对应多组欧拉角（欧拉角不唯一），这里返回其中一组规范解。
    """
    # 解析欧拉约定：传字符串就查表，传 4 元组就直接用
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # 校验是否合法约定
        firstaxis, parity, repetition, frame = axes

    # 根据约定确定三个轴的下标 i, j, k
    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    # 只取左上 3x3（兼容传进来的是 4x4 仿射矩阵）
    M = np.array(mat, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        # 重复轴约定（如 sxyx）：用 i 行两个分量算第二角
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS4:  # 非退化：正常用 atan2 反解三个角
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:  # 退化（万向锁附近）：第三角置 0，避免数值不稳
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        # 非重复轴约定（如默认 sxyz）：用 i 列两个分量算
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS4:  # 非退化
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:  # 退化
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    # parity（奇约定）需要整体取反；frame（动坐标系）需要交换首末角
    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def quat2mat(q):
    """四元数 -> 旋转矩阵。

    输入：q = (w, x, y, z)，可以未归一化（函数内部会处理）
    输出：3x3 旋转矩阵

    用途：部署 preprocess 里，仿真器给的夹爪朝向是四元数，先转成矩阵再转欧拉角。
    """
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z  # 四元数模长平方
    if Nq < _FLOAT_EPS:  # 几乎是零四元数 -> 视为无旋转，返回单位阵
        return np.eye(3)
    s = 2.0 / Nq  # 缩放因子，顺便完成归一化（允许输入未归一化）
    # 预计算各分量乘积，下面按标准公式拼出旋转矩阵
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    return np.array(
        [
            [1.0 - (yY + zZ), xY - wZ, xZ + wY],
            [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
            [xZ - wY, yZ + wX, 1.0 - (xX + yY)],
        ]
    )


def isrotation(
    R: np.ndarray,
    thresh=1e-6,
) -> bool:
    """检查 R 是否是合法旋转矩阵（即 R^T R ≈ 单位阵）。工具函数，本部署没直接用到。"""
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    iden = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(iden - shouldBeIdentity)
    return n < thresh


def euler2mat(ai, aj, ak, axes="sxyz"):
    """欧拉角 -> 旋转矩阵（mat2euler 的逆操作）。

    输入：(ai, aj, ak) 三个欧拉角（弧度），axes 约定
    输出：3x3 旋转矩阵
    本部署没直接用到，但 euler2quat / 测试会间接用到同类逻辑，保留以保持文件完整。
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # 校验
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    # 处理 frame / parity 约定
    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    # 三个角的 sin/cos 及其组合
    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    # 按约定逐元素填充旋转矩阵
    M = np.eye(3)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj * si
        M[i, k] = sj * ci
        M[j, i] = sj * sk
        M[j, j] = -cj * ss + cc
        M[j, k] = -cj * cs - sc
        M[k, i] = -sj * ck
        M[k, j] = cj * sc + cs
        M[k, k] = cj * cc - ss
    else:
        M[i, i] = cj * ck
        M[i, j] = sj * sc - cs
        M[i, k] = sj * cc + ss
        M[j, i] = cj * sk
        M[j, j] = sj * ss + cc
        M[j, k] = sj * cs - sc
        M[k, i] = -sj
        M[k, j] = cj * si
        M[k, k] = cj * ci
    return M


def euler2axangle(ai, aj, ak, axes="sxyz"):
    """欧拉角 -> 轴角。【部署 postprocess 的核心】

    输入：(ai, aj, ak) 三个欧拉角（弧度）
    输出：(vector, theta) —— 旋转轴单位向量 (3,) 和旋转角标量

    用途：模型预测的旋转动作是欧拉角，但仿真器 step 要的是轴角，用这个转。
    实现：先欧拉角->四元数，再四元数->轴角（复用下面两个函数）。
    """
    return quat2axangle(euler2quat(ai, aj, ak, axes))


def euler2quat(ai, aj, ak, axes="sxyz"):
    """欧拉角 -> 四元数。

    输入：(ai, aj, ak) 三个欧拉角（弧度），axes 约定
    输出：四元数 (w, x, y, z)
    是 euler2axangle 的中间步骤。
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # 校验
        firstaxis, parity, repetition, frame = axes

    # 注意这里下标 +1：四元数 q[0]=w 是实部，q[1:4] 对应 x/y/z
    i = firstaxis + 1
    j = _NEXT_AXIS[i + parity - 1] + 1
    k = _NEXT_AXIS[i - parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    # 半角（四元数用半角的 sin/cos）
    ai = ai / 2.0
    aj = aj / 2.0
    ak = ak / 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    # 按约定填四元数四个分量
    q = np.empty((4,))
    if repetition:
        q[0] = cj * (cc - ss)
        q[i] = cj * (cs + sc)
        q[j] = sj * (cc + ss)
        q[k] = sj * (cs - sc)
    else:
        q[0] = cj * cc + sj * ss
        q[i] = cj * sc - sj * cs
        q[j] = cj * ss + sj * cc
        q[k] = cj * cs - sj * sc
    if parity:
        q[j] *= -1.0

    return q


def quat2axangle(quat, identity_thresh=None):
    """四元数 -> 轴角。

    输入：quat = (w, x, y, z)，可未归一化
    输出：(vector, theta) —— 旋转轴单位向量 (3,) 和旋转角标量
    是 euler2axangle 的最后一步。

    边界处理：
      - 含非有限值（inf/nan）-> 角度返回 nan，轴给任意值 [1,0,0]
      - 几乎无旋转（向量部分≈0）-> 返回零角 + 任意轴 [1,0,0]
    """
    quat = np.asarray(quat)
    Nq = np.sum(quat**2)  # 模长平方
    if not np.isfinite(Nq):  # 输入有 inf/nan
        return np.array([1.0, 0, 0]), float("nan")
    # 确定“接近 0”的阈值
    if identity_thresh is None:
        try:
            identity_thresh = np.finfo(Nq.type).eps * 3
        except (AttributeError, ValueError):  # 不是 numpy 浮点类型
            identity_thresh = _FLOAT_EPS * 3
    if Nq < _FLOAT_EPS**2:  # 几乎零四元数，归一化会不可靠
        return np.array([1.0, 0, 0]), 0.0
    if Nq != 1:  # 未归一化则先归一化
        s = math.sqrt(Nq)
        quat = quat / s
    xyz = quat[1:]  # 向量部分 (x, y, z) 决定旋转轴方向
    len2 = np.sum(xyz**2)
    if len2 < identity_thresh**2:
        # 向量部分≈0 -> 无旋转
        return np.array([1.0, 0, 0]), 0.0
    # 夹住 w 在 [-1, 1]，避免 acos 因浮点误差越界；角度 = 2*acos(w)
    theta = 2 * math.acos(max(min(quat[0], 1), -1))
    return xyz / math.sqrt(len2), theta  # 轴归一化 + 角度


def quat2euler(quaternion, axes="sxyz"):
    """四元数 -> 欧拉角（先转矩阵再转欧拉角的便捷封装）。本部署没直接用到。"""
    return mat2euler(quat2mat(quaternion), axes)
