from GFLocalization import *
from EKF import *
from DR_4DOFAUV_DVLGyro import *
from AUV4DOFSimulatedRobot import *
from Feature import *

class EKF_4DOFAUV_InputVelocityMM_DVLDepthYawOM(GFLocalization, DR_4DOFAUV_DVLGyro, EKF):
    """
    This class implements an EKF localization filter for a 4 DOF AUV using an input velocity motion model  incorporating
    DVL linear velocity measurements, a gyro angular speed measurement, as well as depth and yaw measurements.
    Inherits from GFLocalization because it is a Localization method using Gaussian filtering, and from EKF because it uses an EKF.
    It also inherits from DR_4DOFAUV_DVLGyro to reuse its motion model :meth:`solved_prlab.DR_4DOFAUV_DVLGyro.Localize` and the model input :meth:`solved_prlab.DR_4DOFAUV_DVLGyro.GetInput`.
    """
    def __init__(self, kSteps, robot, *args):
        """
        Constructor.

        :param args: arguments to be passed to the base class constructor
        """
        # xs0 = np.array([[0.0, 0.0, 0.0, 0.0]]).T  # initial simulated robot pose
        # robot = SimulatedRobot(xs0)  # instantiate the simulated robot object

        self.dt = 0.1  # dt is the sampling time at which we iterate the KF
        x0 = np.zeros((4, 1))  # initial state x0=[x y z psi u v w r]^T
        P0 = np.zeros((4, 4))  # initial covariance

        self.t_1 = 0
        self.t = 0
        self.Dt = self.t - self.t_1

        index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("z", 2, 0), IndexStruct("yaw", 3, 1)]


        super().__init__(index, kSteps, robot, x0, P0, *args)

        # this is required for plotting


    def f(self, xk_1, uk):  # motion model
        """
        Non-linear motion model using as input the DVL linear velocity and the gyro angular speed:

        .. math::
            x_k&=f(x_{k-1},u_k,w_k) = x_{k-1} \\oplus (u_k + w_k) \\Delta t \\\\
            x_{k-1}&=[x_{k_1}^T, y_{k_1}^T, z_{k_1}^T, \\psi_{k_1}^T]^T\\\\
            u_k&=[u_k, v_k, w_k, r_k]^T

        :param xk_1: previous mean state vector (:math:`x_{k-1}=[x_{k-1}^T, y_{k-1}^T, z_{k-1}^T, \\psi_{k-1}^T]^T`) containing the robot position and heading in the N-Frame
        :param uk: input vector :math:`u_k=[u_k^T, v_k^T, w_k^T, r_k^T]^T` containing the DVL linear velocity and the gyro angular speed, both referenced in the B-Frame
        :return: current mean state vector containing the current robot position and heading (:math:`x_k=[x_k^T, y_k^T, z_k^T, \\psi_k^T]^T`) represented in the N-Frame
        """
        self.t = self.k * self.dt

        self.Dt = self.t - self.t_1

        self.etak_1 = Pose4D(xk_1)  # extract position and heading
        self.nuk = uk  # extract velocity and angular velocity
        xk_bar = self.etak_1.oplus(self.nuk * self.Dt)  # compute state prediction

        self.t_1 = self.t
        return xk_bar

    def Jfx(self, xk_1):
        """
        Jacobian of the motion model with respect to the state vector:

        .. math::
            J_{fx}=\\frac{\\partial f(x_{k-1},u_k,w_k)}{\\partial x_{k-1}} = \\frac{\\partial x_{k-1} \\oplus (u_k + w_k)}{\\partial x_{k-1}} = J_{1\\oplus}
            :label: eq-Jfx-EKF_4DOFAUV_VelocityMM_DVLDepthYawOM

        :param xk_1: Linearization point. By default the linearization point is the previous state vector taken from a class attribute.
        :return: Jacobian matrix
        """
        return Pose4D(xk_1).J_1oplus(self.nuk * self.Dt)

    def Jfw(self, xk_1):
        """
        Jacobian of the motion model with respect to the motion model noise vector:

        .. math::
            J_{fx}=\\frac{\\partial f(x_{k-1},u_k,w_k)}{\\partial w_k} = \\frac{\\partial x_{k-1} \\oplus (u_k + w_k)}{\\partial w_k} = J_{2\\oplus}
            :label: eq-Jfw-EKF_4DOFAUV_VelocityMM_DVLDepthYawOM

        :param xk_1: Linearization point. By default the linearization point is the previous state vector taken from a class attribute.
        :return: Jacobian matrix
        """
        return Pose4D(xk_1).J_2oplus()*self.Dt

    def h(self, xk_bar):
        etak_bar = xk_bar[0:4]  # extract position and heading
        h_depth = etak_bar[2, 0]  # extract depth
        h_yaw = etak_bar[3, 0]  # extract heading

        h = np.zeros((0, 1))
        if self.depth: h = np.block([[h], [h_depth]])
        if self.yaw: h = np.block([[h], [h_yaw]])

        return h

    def Jhmx(self, xk):
        """
        Jacobian of the measurement model with respect to the state vector:

        .. math::
            J_{hmx}=H_{m_k}=\\frac{\\partial h_m(x_k,v_k)}{\\partial x_k} = \\frac{\\partial [z_{depth}^T, \\psi_{compass}^T]^T}{\\partial x_k}
            =\\begin{bmatrix} 0 & 0 & 1 & 0 \\\\ 0 & 0 & 0 & 1 \\end{bmatrix}
            :label: eq-Jhmx-EKF_4DOFAUV_VelocityMM_DVLDepthYawOM

        :param xk: mean state vector containing the robot position and heading (:math:`x_k=[x_k^T, y_k^T, z_k^T, \\psi_k^T]^T`) represented in the N-Frame
        :return: observation matrix (Jacobian) matrix eq. :eq:`eq-Jhmx-EKF_4DOFAUV_VelocityMM_DVLDepthYawOM`.
        """
        Hk = np.array([[0, 0, 1, 0],
                       [0, 0, 0, 1]])
        return Hk

    def Jhmv(self, xk_bar):
        """
        Jacobian of the measurement model with respect to the measurement noise vector:

        .. math::
            J_{hmv}=V_{m_k}=\\frac{\\partial h_m(x_k,v_k)}{\\partial v_k} = I_{2 \\times 2}
            :label: eq-Jhmv-EKF4DOFAUVVelocityMMDVLDepthYawOM

        :param xk: mean state vector containing the robot position and heading (:math:`x_k=[x_k^T, y_k^T, z_k^T, \\psi_k^T]^T`) represented in the N-Frame
        :return: observation noise (Jacobian) matrix eq. :eq:`eq-Jhmv-EKF4DOFAUVVelocityMMDVLDepthYawOM`.
        """
        return np.eye(2)

    def GetInput(self):
        """
        This method calls the :meth:DR_4DOFAUV_DVLGyro.DR_4DOFAUV_DVLGyro.GetInput` method to get the robot velocity  and its uncertainty.
        :return: [uk,Qk]: robot velocity and its noise covariance matrix
        """
        uk,Qk=DR_4DOFAUV_DVLGyro.GetInput(self)

        return uk,Qk


    def GetMeasurements(self):  # override the observation model
        """

        :return:
        """
        zk = np.zeros((0, 1))  # empty vector
        Rk = np.zeros((0, 0))  # empty matrix
        Hm = np.zeros((0, 4))  # empty vector
        Vk = np.zeros((0, 0))  # empty matrix

        self.depth = False
        self.yaw = False
        self.dvl = False
        z_depth, R_depth = self.robot.ReadDepth()
        if z_depth.size > 0:  # if there is a measurement
            H_depth=np.array([[0, 0, 1, 0]])
            V_depth=np.eye(1)
            zk, Rk = np.block([[zk], [z_depth]]), scipy.linalg.block_diag(Rk, R_depth)
            Hm, Vk = np.block([[Hm], [H_depth]]), scipy.linalg.block_diag(Vk, V_depth)
            self.depth = True

        z_yaw, sigma2_yaw = self.robot.ReadCompass()
        if z_yaw.size > 0:  # if there is a measurement
            H_yaw=np.array([[0, 0, 0, 1]])
            V_yaw=np.eye(1)
            zk, Rk = np.block([[zk], [z_yaw]]), scipy.linalg.block_diag(Rk, sigma2_yaw)
            Hm, Vk = np.block([[Hm], [H_yaw]]), scipy.linalg.block_diag(Vk, V_yaw)
            self.yaw = True

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

    xs0 = np.zeros((8, 1))
    kSteps = 5000
    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)]

    robot = AUV4DOFSimulatedRobot(xs0, M)  # instantiate the simulated robot object

    x0 = Pose4D(np.zeros((4, 1)))

    dr_robot = DR_4DOFAUV_DVLGyro(index, kSteps, robot, x0)

    x0 = np.zeros((4, 1))
    P0 = np.zeros((4, 4))

    auv = EKF_4DOFAUV_InputVelocityMM_DVLDepthYawOM(kSteps, robot)  # initialize robot and KF

    auv.LocalizationLoop(x0, P0, np.array([[0.5, 0.0, 0.0, 0.03]]).T)

    exit(0)