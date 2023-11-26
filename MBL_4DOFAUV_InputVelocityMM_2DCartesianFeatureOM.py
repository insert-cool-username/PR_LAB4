from MapFeature import *
from FEKFMBL import *
from EKF_4DOFAUV_InputVelocityMM_DVLDepthYawOM import *
from Pose import *

class MBL_4DOFAUV_InputVelocityMM_2DCartesianFeatureOM(Cartesian2DMapFeature, FEKFMBL, EKF_4DOFAUV_InputVelocityMM_DVLDepthYawOM):
    """
    Feature EKF Map based Localization of a 4 DOF AUV (:math:`x_k=[^Nx_{B_k} ~^Ny_{B_k} ~^Nz_{B_k} ~^N\\psi_{B_k} ~]^T`) using a 2D Cartesian feature map (:math:`M=[[^Nx_{F_1} ~^Ny_{F_1}] ~[x_{F_2} ~^Ny_{F_2}] ~... ~[^Nx_{F_n} ~^Ny_{F_n}]]^T`),
    and an input velocity motion model (:math:`u_k=[^Bu_k ~^Bv_k ~^Bw_k ~^Br_k]^T`). The linear velocity is measured by a DVL (:math:`z_k=[^Bv_k ~^Bw_k]^T`), the depth is measured by a depth sensor (:math:`z_k=[^Nz_{B_k}]^T`), and the yaw is measured by a compass (:math:`z_k=[^N\\psi_{B_k}]^T`).
    """
    def __init__(self, *args):
        self.Feature = globals()["CartesianFeature"]
        self.Pose = globals()["Pose4D"]
        super().__init__(*args)

if __name__ == '__main__':

    M = [CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.

    # M = [CartesianFeature(np.array([[-40, 5, 0]]).T),
    #        CartesianFeature(np.array([[-5, 40, 0]]).T),
    #        CartesianFeature(np.array([[-5, 25, 0]]).T),
    #        CartesianFeature(np.array([[-3, 50, 0]]).T),
    #        CartesianFeature(np.array([[-20, 3, 0]]).T),
    #        CartesianFeature(np.array([[40, -40, 0]]).T)]  # feature map. Position of 3 point features in the world frame.

    xs0 = np.zeros((8, 1))
    kSteps = 5000
    alpha = 0.95

    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)]

    robot = AUV4DOFSimulatedRobot(xs0, M)  # instantiate the simulated robot object

    x0 = Pose4D(np.zeros((4, 1)))
    dr_robot = DR_4DOFAUV_DVLGyro(index, kSteps, robot, x0)
    robot.SetMap(M)

    auv = MBL_4DOFAUV_InputVelocityMM_2DCartesianFeatureOM(M, alpha, kSteps, robot)

    P0 = np.zeros((4, 4))
    usk=np.array([[0.5, 0, 0, 0.03]]).T
    auv.LocalizationLoop(x0, P0, usk)

    exit(0)
