from GFLocalization import *
from EKF import *
from DR_4DOFAUV_DVLGyro import *
from AUV4DOFSimulatedRobot import *
from Feature import *

class EKF_4DOFAUV_CtVelocityMM_DVLDepthYawOM(GFLocalization,DR_4DOFAUV_DVLGyro,EKF):
    def __init__(self, kSteps, robot, *args):
        # this is required for plotting
        x0 = np.zeros((8, 1))  # initial state x0=[x y z psi u v w r]^T
        P0 = np.zeros((8, 8))  # initial covariance

        index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None),  IndexStruct("z", 2, 0), IndexStruct("yaw", 3, 1),
                 IndexStruct("u", 4, 2), IndexStruct("v", 5, 3),  IndexStruct("w", 6, 0), IndexStruct("yaw_dot", 7, None)]

        super().__init__(index, kSteps, robot, x0, P0, *args)

    def f(self, xk_1, uk):
        etak_1 = Pose4D(xk_1[0:4])  # extract position and heading
        nuk_1 = xk_1[4:8]  # extract velocity and angular velocity

        xk_bar = np.block([[etak_1.oplus(nuk_1 * self.dt)],
                           [nuk_1]])  # compute state prediction
        return xk_bar

    def Jfx(self, xk_1):
        etak_1 = Pose4D(xk_1[0:4])  # extract position and heading
        nuk_1 = xk_1[4:8]  # extract velocity and angular velocity
        J = np.block([[etak_1.J_1oplus(nuk_1 * self.dt), etak_1.J_2oplus() * self.dt],
                      [np.zeros((4, 4)), np.eye(4)]])
        return J

    def Jfw(self, xk_1):
        etak_1 = Pose4D(xk_1[0:4])  # extract position and heading

        J = np.block([[etak_1.J_2oplus() * 0.5 * self.dt ** 2],
                      [np.eye(4) * self.dt]])
        return J

    def h(self, xk_bar):
        etak_bar = xk_bar[0:4]  # extract position and heading
        nuk_bar = xk_bar[4:8]  # extract velocity and angular velocity
        h_depth = etak_bar[2, 0]  # extract depth
        h_yaw = etak_bar[3, 0]  # extract heading
        h_dvl = nuk_bar[0:3, 0].reshape(3, 1)  # extract linear velocity

        h = np.zeros((0, 1))
        if self.depth: h = np.block([[h], [h_depth]])
        if self.yaw: h = np.block([[h], [h_yaw]])
        if self.dvl: h = np.block([[h], [h_dvl]])

        return h

    def Jhmx(self, xk_bar):
        H_depth = np.array([[0, 0, 1, 0, 0, 0, 0, 0]])
        H_yaw = np.array([[0, 0, 0, 1, 0, 0, 0, 0]])
        H_dvl = np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0]])

        Hk = np.zeros((0, 8))  # empty matrix
        if self.depth: Hk = np.block([[Hk], [H_depth]])
        if self.yaw: Hk = np.block([[Hk], [H_yaw]])
        if self.dvl: Hk = np.block([[Hk], [H_dvl]])

        return Hk

    def Jhmv(self, xk_bar):
        V_depth = np.eye(1)
        V_yaw = np.eye(1)
        V_dvl = np.eye(3)

        Vk = np.zeros((0, 0))  # empty matrix
        if self.depth: Vk = scipy.linalg.block_diag(Vk, V_depth)
        if self.yaw: Vk = scipy.linalg.block_diag(Vk, V_yaw)
        if self.dvl: Vk = scipy.linalg.block_diag(Vk, V_dvl)

        return Vk


    def GetInput(self):
        """

        :return:
        """
        uk = np.zeros((4, 1))
        # Covariance of the motion model computed from the covariance of the acceleration noise
        self.Qk = self.robot.Qsk

        return uk, self.Qk

    def GetMeasurements(self):  # override the observation model
        """

        :return:
        """
        zk = np.zeros((0, 1))  # empty vector
        Rk = np.zeros((0, 0))  # empty matrix
        Hm = np.zeros((0, 8))  # empty vector
        Vk = np.zeros((0, 0))  # empty matrix

        self.depth = False
        self.yaw = False
        self.dvl = False
        z_depth, R_depth = self.robot.ReadDepth()
        if z_depth.size > 0:  # if there is a measurement
            H_depth=np.array([[0, 0, 1, 0, 0, 0, 0, 0]])
            V_depth=np.eye(1)
            zk, Rk = np.block([[zk], [z_depth]]), scipy.linalg.block_diag(Rk, R_depth)
            Hm, Vk = np.block([[Hm], [H_depth]]), scipy.linalg.block_diag(Vk, V_depth)
            self.depth = True

        z_yaw, sigma2_yaw = self.robot.ReadCompass()
        if z_yaw.size > 0:  # if there is a measurement
            H_yaw=np.array([[0, 0, 0, 1, 0, 0, 0, 0]])
            V_yaw=np.eye(1)
            zk, Rk = np.block([[zk], [z_yaw]]), scipy.linalg.block_diag(Rk, sigma2_yaw)
            Hm, Vk = np.block([[Hm], [H_yaw]]), scipy.linalg.block_diag(Vk, V_yaw)
            self.yaw = True

        z_dvl, Q_dvl = self.robot.ReadDVL()
        if z_dvl.size > 0:
            H_dvl = np.array([[0, 0, 0, 0, 1, 0, 0,0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0]])
            V_dvl = np.eye(3)
            zk, Rk = np.block([[zk], [z_dvl]]), scipy.linalg.block_diag(Rk, Q_dvl)
            Hm, Vk = np.block([[Hm], [H_dvl]]), scipy.linalg.block_diag(Vk, V_dvl)
            self.dvl = True

        Hk=np.zeros((Hm.shape[0],self.xk_bar.T.shape[1]))
        (r,c)=Hm.shape
        Hk[0:r,0:c]=Hm

        return zk, Rk, Hk, Vk

if __name__ == '__main__':

    M = [CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.

    kSteps = 5000
    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)]

    xs0 = np.zeros((8, 1))
    robot = AUV4DOFSimulatedRobot(xs0, M)  # instantiate the simulated robot object
    x0 = Pose4D(np.zeros((4, 1)))
    dr_robot = DR_4DOFAUV_DVLGyro(index, kSteps, robot, x0)
    x0 = np.zeros((8, 1))
    P0 = np.zeros((8, 8))
    auv = EKF_4DOFAUV_CtVelocityMM_DVLDepthYawOM(kSteps, robot)  # initialize robot and KF
    auv.LocalizationLoop(x0, P0, np.array([[0.5, 0.0, 0.0, 0.03]]).T)

    exit(0)