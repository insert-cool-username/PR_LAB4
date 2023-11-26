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

        # TODO: To decide the best place to do this.
        # self.xB_dim = 3  # dimension of the robot state vector
        # self.xBPose_dim = 3  # dimension of the robot pose within the state vector

        super().__init__(self.index, kSteps, robot, self.x0, self.P0, *args) # this line must appear here. do not move it.

        # Converts from encoder pulses to linear and angular velocities
        # [u,v]^T= Ke · [nL, nR]^T
        self.Kn = np.array([[np.pi * robot.wheelRadius / (robot.pulse_x_wheelTurns * robot.dt),
                        np.pi * robot.wheelRadius / (robot.pulse_x_wheelTurns * robot.dt)],
                       [-2 * np.pi * robot.wheelRadius / (robot.pulse_x_wheelTurns * robot.wheelBase * robot.dt),
                        2 * np.pi * robot.wheelRadius / (robot.pulse_x_wheelTurns * robot.wheelBase * robot.dt)]])

        # converts from linear and angular velocities to encoder pulses
        # [nL, nR]^T = Ke_inv · [u,r]^T
        self.Kn_inv = np.linalg.inv(self.Kn)

    def f(self, xk_1, uk):
        etak_1 = Pose3D(xk_1[0:3])  # extract position and heading
        nuk_1 = xk_1[3:6]  # extract velocity and angular velocity

        xk_bar = np.block([[etak_1.oplus(nuk_1 * self.dt)],
                           [nuk_1]])  # compute state prediction
        return xk_bar

    def Jfx(self, xk_1):
        etak_1 = Pose3D(xk_1[0:3])  # extract position and heading
        nuk_1 = xk_1[3:6]  # extract velocity and angular velocity
        J = np.block([[etak_1.J_1oplus(nuk_1 * self.dt), etak_1.J_2oplus() * self.dt],
                      [np.zeros((3, 3)), np.eye(3)]])
        return J

    def Jfw(self, xk_1):
        etak_1 = Pose3D(xk_1[0:3])  # extract position and heading

        J = np.block([[etak_1.J_2oplus() * 0.5 * self.dt ** 2],
                      [np.eye(3) * self.dt]])
        return J

    def h(self,xk):#:hm(self, xk):
        h_yaw = xk[2,0]  # extract pose
        nu = np.array([xk[3,0],xk[5,0]]).reshape(2,1) # extract velocity
        h_n = self.Kn_inv @ nu

        h=np.zeros((0, 1))

        if self.yaw:
            h=np.block([[h],[h_yaw]])
        if self.vel:
            h=np.block([[h],[h_n]])

        return h  # return heading measurement (yaw) & linear velocity

    def GetInput(self):
        """

        :return:
        """
        uk = np.zeros((3, 1))
        # Covariance of the motion model computed from the covariance of the acceleration noise
        self.Qk = self.robot.Qsk

        return uk, self.Qk

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