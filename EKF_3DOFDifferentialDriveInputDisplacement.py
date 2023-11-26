from GFLocalization import *
from EKF import *
from DR_3DOFDifferentialDrive import *
from DifferentialDriveSimulatedRobot import *

class EKF_3DOFDifferentialDriveInputDisplacement(GFLocalization, DR_3DOFDifferentialDrive, EKF):
    """
    This class implements an EKF localization filter for a 3 DOF Diffenteial Drive using an input displacement motion model incorporating
    yaw measurements from the compass sensor.
    It inherits from :class:`GFLocalization.GFLocalization` to implement a localization filter, from the :class:`DR_3DOFDifferentialDrive.DR_3DOFDifferentialDrive` class and, finally, it inherits from
    :class:`EKF.EKF` to use the EKF Gaussian filter implementation for the localization.
    """
    def __init__(self, kSteps, robot, *args):
        """
        Constructor.

        :param args: arguments to be passed to the base class constructor
        """
        # xs0 = np.array([[0.0, 0.0, 0.0, 0.0]]).T  # initial simulated robot pose
        # robot = SimulatedRobot(xs0)  # instantiate the simulated robot object

        self.dt = 0.1  # dt is the sampling time at which we iterate the KF
        x0 = np.zeros((3, 1))  # initial state x0=[x y z psi u v w r]^T
        P0 = np.zeros((3, 3))  # initial covariance

        # this is required for plotting
        index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("z", 2, 0), IndexStruct("yaw", 3, 1)]

        self.t_1 = 0
        self.t = 0
        self.Dt = self.t - self.t_1
        super().__init__(index, kSteps, robot, x0, P0, *args)

    def GetInput(self):
        """
        Calls the :meth:`DR_3DOFDifferentialDrive.DR_3DOFDifferentialDrive.GetInput` method from the :class:`DR_3DOFDifferentialDrive.DR_3DOFDifferentialDrive` class to get the
        robot displacement. Then it computes the uncertainty of the robot displacement from the covariance matrix of the encoders.
        Finally, it returns the robot displacement and the uncertainty of the robot displacement.

        :return: uk,Qk: robot displacemnt and the covariance of the robot displacement.
        """
        uk=DR_3DOFDifferentialDrive.GetInput(self)

        K = (np.pi * self.wheelRadius / self.robot.pulse_x_wheelTurns) * np.array([[1,1],[0,0],[1/self.wheelBase,-1/self.wheelBase]])

        Qk= K @ self.robot.Re @ K.T

        return uk,Qk

    def f(self, xk_1, uk):  # motion model
        """
        Non-linear motion model using as input the robot displacement:

        .. math::
            ^N \\hat {\\bar x}_k&=f(^N\\hat x_{k-1},^Bu_k) = {}^Nx_{k-1} \\oplus ^Bu_k \\\\
            {}^Nx_{k-1}&=[^Nx_{k_1}^T~ ^Ny_{k_1}^T~ ^N\\psi_{k_1}^T]^T\\\\
            ^Bu_k&=^B[\Delta x_k ~\Delta y_k ~\Delta \psi_k]^T

        :param xk_1: previous mean state vector (:math:`x_{k-1}=[x_{k-1}^T, y_{k-1}^T,\\psi_{k-1}^T]^T`) containing the robot position and heading in the N-Frame
        :param uk: input vector :math:`u_k=[\Delta x_k ~\Delta y_k ~\Delta \psi_k]^T` containing the robot displacement referenced in the B-Frame
        :return: xk_bar: predicted mean state vector containing the current robot position and heading (:math:`\\bar x_k=[x_k^T, y_k^T, \\psi_k^T]^T`) represented in the N-Frame
        """
        self.t = self.k * self.dt
        self.Dt = self.t - self.t_1

        self.etak_1 = Pose3D(xk_1)  # extract position and heading
        self.uk = uk  # extract velocity and angular velocity
        self.xk_bar = self.etak_1.oplus(self.uk)  # compute state prediction

        self.t_1 = self.t
        return self.xk_bar

    def Jfx(self, xk_1):
        """
        Jacobian of the motion model with respect to the state vector:

        .. math::
            J_{fx}=\\frac{\\partial f(x_{k-1},u_k,w_k)}{\\partial x_{k-1}} = \\frac{\\partial x_{k-1} \\oplus (u_k + w_k)}{\\partial x_{k-1}} = J_{1\\oplus}
            :label: eq-Jfx-EKF_3DOFDifferentialDriveInputDisplacement

        :param xk_1: Linearization point. By default the linearization point is the previous state vector taken from a class attribute.
        :return: Jacobian matrix
        """
        etak_1 = Pose3D(xk_1)
        return etak_1.J_1oplus(self.uk)

    def Jfw(self, xk_1):
        """
        Jacobian of the motion model with respect to the motion model noise vector:

        .. math::
            J_{fx}=\\frac{\\partial f(x_{k-1},u_k,w_k)}{\\partial w_k} = \\frac{\\partial x_{k-1} \\oplus (u_k + w_k)}{\\partial w_k} = J_{2\\oplus}
            :label: eq-Jfw-EKF_3DOFDifferentialDriveInputDisplacement

        :param xk_1: Linearization point. By default the linearization point is the previous state vector taken from a class attribute.
        :return: Jacobian matrix
        """
        etak_1 = Pose3D(xk_1)
        return etak_1.J_2oplus()
    def h(self, xk):
        if self.k % self.robot.yaw_reading_frequency == 0:
            return xk[2]  # return heading measurement (yaw)
        else:
            return np.zeros((0,1))

    # def Jhmx(self, xk):
    #     """
    #     Jacobian of the measurement model with respect to the state vector:
    #
    #     .. math::
    #         J_{hmx}=H_{m_k}=\\frac{\\partial h_m(x_k,v_k)}{\\partial x_k} = \\frac{\\partial [z_{depth}^T, \\psi_{compass}^T]^T}{\\partial x_k}
    #         =\\begin{bmatrix} 0 & 0 & 1 & 0 \\\\ 0 & 0 & 0 & 1 \\end{bmatrix}
    #         :label: eq-Jhmx-EKF_3DOFDifferentialDriveInputDisplacement
    #
    #     :param xk: mean state vector containing the robot position and heading (:math:`x_k=[x_k^T, y_k^T, z_k^T, \\psi_k^T]^T`) represented in the N-Frame
    #     :return: observation matrix (Jacobian) matrix eq. :eq:`eq-Jhmx-EKF_3DOFDifferentialDriveInputDisplacement`.
    #     """
    #     Hk = np.array([[0, 0, 1]])
    #     return Hk
    #
    # def Jhmv(self, xk_bar):
    #     """
    #     Jacobian of the measurement model with respect to the measurement noise vector:
    #
    #     .. math::
    #         J_{hmv}=V_{m_k}=\\frac{\\partial h_m(x_k,v_k)}{\\partial v_k} = I_{2 \\times 2}
    #         :label: eq-Jhmv-EKF_3DOFDifferentialDriveInputDisplacement
    #
    #     :param xk: mean state vector containing the robot position and heading (:math:`x_k=[x_k^T, y_k^T, z_k^T, \\psi_k^T]^T`) represented in the N-Frame
    #     :return: observation noise (Jacobian) matrix eq. :eq:`eq-Jhmv-EKF_3DOFDifferentialDriveInputDisplacement`.
    #     """
    #     return np.eye(1)

    def GetMeasurements(self):  # override the observation model
        """
        Gets the measurement vector and the measurement noise covariance matrix from the robot. The measurement vector contains the depth read from the depth sensor and the heading read from the compass sensor.

        .. math::
            z_k&=\\begin{bmatrix} z_{depth}^T & \\psi_{compass}^T \\end{bmatrix}^T\\\\
            R_k&=\\begin{bmatrix} \\sigma_{depth}^2 & 0 \\\\ 0 & \\sigma_{compass}^2 \\end{bmatrix}
            :label: eq-zk-EKF_3DOFDifferentialDriveInputDisplacement

        :return: observation vector :math:`z_k` and observation noise covariance matrix :math:`R_k` defined in eq. :eq:`eq-zk-EKF_3DOFDifferentialDriveInputDisplacement`.
        """

        z_yaw, sigma2_yaw = self.robot.ReadCompass()
        Hk = np.array([[0, 0, 1]])
        Vk = np.eye(1)

        return z_yaw, sigma2_yaw, Hk, Vk

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

    x0 = np.zeros((3, 1))
    P0 = np.zeros((3, 3))

    dd_robot = EKF_3DOFDifferentialDriveInputDisplacement(kSteps,robot)  # initialize robot and KF
    dd_robot.LocalizationLoop(x0, P0, np.array([[0.5, 0.03]]).T)

    exit(0)