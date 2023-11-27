from GFLocalization import *
from EKF import *
from DR_3DOFDifferentialDrive import *
from DifferentialDriveSimulatedRobot import *

class EKF_3DOFDifferentialDriveCtVelocity(GFLocalization, DR_3DOFDifferentialDrive, EKF):

    def __init__(self, kSteps, robot, *args):

        self.x0 = np.zeros((6, 1))  # initial state x0=[x y z psi u v w r]^T
        self.P0 = np.zeros((6, 6))  # initial covariance

        # this is required for plotting
        self.index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1),
                 IndexStruct("u", 3, 2), IndexStruct("v", 4, 3), IndexStruct("yaw_dot", 5, None)]

        # TODO: To be completed by the student

    def f(self, xk_1, uk):
        # TODO: To be completed by the student

        return xk_bar

    def Jfx(self, xk_1):
        # TODO: To be completed by the student

        return J

    def Jfw(self, xk_1):
        # TODO: To be completed by the student

        return J

    def h(self, xk):  #:hm(self, xk):
        # TODO: To be completed by the student

        return h  # return the expected observations

    def GetInput(self):
        """

        :return: uk,Qk:
        """
        # TODO: To be completed by the student

        return uk, Qk

    def GetMeasurements(self):  # override the observation model
        """

        :return: zk, Rk, Hk, Vk
        """
        # TODO: To be completed by the student

        return zk, Rk, Hk, Vk
    def GetMeasurements(self):  # override the observation model
        """

        :return:
        """
        zk = np.zeros((0, 1))  # empty vector
        Rk = np.zeros((0, 0))  # empty matrix

        self.yaw = False
        self.vel = False

        Hk, Vk = np.zeros((0,6)), np.zeros((0,0))
        H_yaw, V_yaw = np.array([[0,0,1,0,0,0]]), np.eye(1)

        z_yaw, sigma2_yaw = self.robot.ReadCompass()

        if z_yaw.size>0:  # if there is a measurement
            zk, Rk = np.block([[zk], [z_yaw]]), scipy.linalg.block_diag(Rk, sigma2_yaw)
            Hk, Vk = np.block([[Hk], [H_yaw]]), scipy.linalg.block_diag(Vk, V_yaw)
            self.yaw = True

        n, Rn = self.robot.ReadEncoders(); L=0; R=1  # read sensors

        if n.size>0:  # if there is a measurement
            self.vel=True

            H_n= np.array([[ 0,0,0,self.Kn_inv[0,0],0,self.Kn_inv[0,1]],
                            [ 0,0,0,self.Kn_inv[1,0],0,self.Kn_inv[1,1]]])

            zk, Rk = np.block([[zk], [n]]), scipy.linalg.block_diag(Rk, Rn)
            Hk, Vk = np.block([[Hk], [H_n]]), scipy.linalg.block_diag(Vk, np.eye(2))

        if zk.shape[0] == 0: # if there is no measurement
            return np.zeros((1,0)), np.zeros((0,0)),np.zeros((1,0)), np.zeros((0,0))
        else:
            return zk, Rk, Hk, Vk

if __name__ == '__main__':

    M = [CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.

    xs0 = np.zeros((6,1))  # initial simulated robot pose

    robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object
    kSteps = 5000

    xs0 = np.zeros((6, 1))  # initial simulated robot pose
    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)]

    x0 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
    P0 = np.diag(np.array([0.0, 0.0, 0.0, 0.5 ** 2, 0 ** 2, 0.05 ** 2]))

    dd_robot = EKF_3DOFDifferentialDriveCtVelocity(kSteps, robot)  # initialize robot and KF
    dd_robot.LocalizationLoop(x0, P0, np.array([[0.5, 0.03]]).T)  # run localization loop

    exit(0)