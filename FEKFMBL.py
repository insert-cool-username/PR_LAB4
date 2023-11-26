import scipy
from GFLocalization import *
from MapFeature import *
from EKF import *
import math
from blockarray import *

class FEKFMBL(GFLocalization, MapFeature):
    """
    Feature Extended Kalman Filter Map based Localization class. Inherits from :class:`GFLocalization.GFLocalization` and :class:`MapFeature.MapFeature`.
    The first one provides the basic functionality of a localization algorithm, while the second one provides the basic functionality required to use features.
    :class:`FEKFMBL.FEKFMBL` extends those classes by adding functionality to use a map based on features.
    """
    xB = -2  # constant used to index the state vector and the covariance matrix, to select the robot state
    x_eta = -1  # constant used to index the state vector and the covariance matrix, to select the robot pose
    def __init__(self, M, alpha,  *args):
        """
        Constructor of the FEKFMBL class.

        :param xBpose_dim: dimensionality of the robot pose within the state vector
        :param xB_dim: dimensionality of the state vector
        :param xF_dim: dimentsionality of a feature
        :param zfi_dim: dimensionality of a single feature observation
        :param M: Feature Based Map :math:`M =[^Nx_{F_1}^T~...~^Nx_{F_{n_f}}^T]^T`
        :param alpha: Chi2 tail probability. Confidence interaval of the individual compatibility test
        :param args: arguments to be passed to the EKFLocalization constructor
        """

        super().__init__(*args)  # initialize EKFLocalization
        self.xBpose_dim = self.Pose().shape[0] # Robot Pose dimensionality
        self.xB_dim = self.xk_1.shape[0]  # Robot State dimensionality (might include the velocity or other terms)
        self.xF_dim = self.Feature.feature.shape[0]  # Feature dimensionality
        self.zfi_dim = self.s2o(self.Feature.feature).shape[0]  # dimensionality of a single feature observation

        self.M = M  # Feature Based Map
        self.nf = len(M)  # number of features
        self.alpha = alpha  # Chi2 tail probability - Confidence interval 95%
        self.plt_zf_ellipse = []  # used for plotting the robot ellipse
        self.plt_zf_line = []  # used for plotting the line towards the robot ellipse

        self.plt_robotEllipse, = plt.plot([], [], 'b')
        self.plt_hf_ellipse = []
        self.plt_zf_ellipse = []
        self.plt_zf_line = []
        self.plt_samples = []

    def h(self, xk):  # overloaded to stack measurements and feature observations
        """
        We do differenciate two types of observations:

        * Measurements: :math:`z_m`correspond to observations of the state variable (position, velocity, etc...)
        * Feature Observations: :math:`z_f` correspond to observations of the features (CartesianFeature, PolarFeature, EsphericalFeature, etc...).

        This method implements the full observation model including the measurements and feature observations:

        .. math::
            z_k = h(x_k,v_k) \\Rightarrow \\begin{bmatrix} z_m \\\\ z_f \\end{bmatrix} = \\begin{bmatrix} h_m(x_k,v_m) \\\\ h_f(x_k,v_f) \\end{bmatrix} ~;~ v_k=[v_m^T ~v_f^T]^T
            :label: eq-mblh

        This method calls :meth:`h_m` to compute the expected measurements and  the :meth:`MapFeature.MapFeature.hf` method to compute the expected feature observations.
        The method returns an stacked vector of expected measurements and feature observations.

        :param xk: mean state vector used as linearization point
        :return: Joint stacked vector of the expected mesurement and feature observations
        """
        h_m = self.hm(xk)
        h_f = self.hf(xk)

        h_mf = np.block([[h_m], [h_f]])  # if (h_m is not None and h_f is not None) else h_m if h_m is not None else h_f
        return h_mf

    def hm(self,xk):
        """
        Measurement observation model. This method computes the expected measurements :math:`h_m(x_k,v_m)` given the
        mean state vector :math:`x_k` and the measurement noise :math:`v_m`. It is implemented by calling to the ancestor
        class :meth:`EKF.EKF.h` method.

        :param xk: mean state vector.
        :return: expected measruments.
        """

        return super().h(xk)

    def SquaredMahalanobisDistance(self, hfj, Pfj, zfi, Rfi):
        """
        Computes the squared Mahalanobis distance between the expected feature observation :math:`hf_j` and the feature observation :math:`z_{f_i}`.

        :param hfj: expected feature observation
        :param Pfj: expected feature observation covariance
        :param zfi: feature observation
        :param Rfi: feature observation covariance
        :return: Squared Mahalanobis distance between the expected feature observation :math:`hf_j` and the feature observation :math:`z_{f_i}`
        """
        nu_ij = zfi - hfj  # Compute the innovation
        S_ij = Rfi + Pfj  # Compute the innovation covariance
        D2_ij = float(nu_ij.T @ np.linalg.inv(S_ij) @ nu_ij)  # Compute the Mahalanobis distance
        return D2_ij

    def IndividualCompatibility(self, D2_ij, dof, alpha):
        """
        Computes the individual compatibility test for the squared Mahalanobis distance :math:`D^2_{ij}`. The test is performed using the Chi-Square distribution with :math:`dof` degrees of freedom and a significance level :math:`\\alpha`.

        :param D2_ij: squared Mahalanobis distance
        :param dof: number of degrees of freedom
        :param alpha: confidence level
        :return: bolean value indicating if the Mahalanobis distance is smaller than the threshold defined by the confidence level
        """
        return D2_ij < scipy.stats.chi2.ppf(alpha, dof)

    def ICNN(self, hf, Phf, zf, Rf, dim):
        """
        Individual Compatibility Nearest Neighbor (ICNN) data association algorithm. Given a set of expected feature
        observations :math:`h_f` and a set of feature observations :math:`z_f`, the algorithm returns a pairing hypothesis
        :math:`H` that associates each feature observation :math:`z_{f_i}` with the expected feature observation
        :math:`h_{f_j}` that minimizes the Mahalanobis distance :math:`D^2_{ij}`.

        :param hf: vector of expected feature observations
        :param Phf: Covariance matrix of the expected feature observations
        :param zf: vector of feature observations
        :param Rf: Covariance matrix of the feature observations
        :param dim: feature dimensionality
        :return: The vector of asociation hypothesiss
        """
        hf=BlockArray(hf,dim); Phf=BlockArray(Phf,dim)
        zf=BlockArray(zf,dim); Rf=BlockArray(Rf,dim)

        nhf = hf.size // dim # number of expected feature observations
        nzf = zf.size // dim # number of features observed
        Hp = []  # initialize the pairing Hypotessis list

        for Fj in range(nzf):  # for each feature observation
            nearest = None
            D2_ij_min = math.inf  # initialize the minimum Mahalanobis distance
            for Fi in range(nhf):  # for each feature in hf
                xFi, PFi = hf[[Fi]], Phf[[Fi,Fi]]
                zFj, RFj = zf[[Fj]], Rf[[Fj,Fj]]
                D2_ij = self.SquaredMahalanobisDistance(xFi, PFi, zFj, RFj)
                if self.IndividualCompatibility(D2_ij, dim, self.alpha) and D2_ij < D2_ij_min:
                    nearest = Fi
                    D2_ij_min = D2_ij

            Hp.append(nearest)
        return Hp

    def DataAssociation(self, xk, Pk, zf, Rf):
        """
        Data association algorithm. Given state vector (:math:`x_k` and :math:`P_k`) including the robot pose and a set of feature observations
        :math:`z_f` and its covariance matrices :math:`R_f`,  the algorithm  computes the expected feature
        observations :math:`h_f` and its covariance matrices :math:`P_f`. Then it calls an association algorithms like
        :meth:`ICNN` (JCBB, etc.) to build a pairing hypothesis associating the observed features :math:`z_f`
        with the expected features observations :math:`h_f`.

        The vector of association hypothesis :math:`H` is stored in the :attr:`H` attribute and its dimension is the
        number of observed features within :math:`z_f`. Given the :math:`j^{th}` feature observation :math:`z_{f_j}`, *self.H[j]=i*
        means that :math:`z_{f_j}` has been associated with the :math:`i^{th}` feature . If *self.H[j]=None* means that :math:`z_{f_j}`
        has not been associated either because it is a new observed feature or because it is an outlier.

        :param xk: mean state vector including the robot pose
        :param Pk: covariance matrix of the state vector
        :param zf: vector of feature observations
        :param Rf: Covariance matrix of the feature observations
        :return: The vector of asociation hypothesiss
        """
        if zf.shape[0]==0:
            return []

        #d=self.zfi_dim # dimension of each feature observation
        hf=BlockArray(np.zeros((self.nf*self.zfi_dim,1)),self.zfi_dim)
        Phf=BlockArray(np.zeros((self.nf*self.zfi_dim,self.nf*self.zfi_dim)),self.zfi_dim)
        for Fj in range(self.nf):    # for mapped features
            h_Fj = self.hfj(xk, Fj)
            J= self.Jhfjx(xk, Fj)
            Ph_Fj = J @ Pk @ J.T
            hf[[Fj]]=h_Fj
            Phf[[Fj,Fj]]=Ph_Fj
            #print("hf[[Fj]]",hf[[Fj]])
            #print(self.IndividualCompatibility(self.SquaredMahalanobisDistance(h_Fj,Ph_Fj,zf[0:2],Rf[0:2,0:2]),2,0.95))

        H = self.ICNN(hf, Phf, zf, Rf,self.zfi_dim) # Individual Compatibility Nearest Neighbor
        return H

    def Localize(self, xk_1, Pk_1):
        """
        Localization iteration. Reads the input of the motion model, performs the prediction step (:meth:`EKF.EKF.Prediction`), reads the measurements
        and the features, solves the data association calling :meth:`DataAssociation` and the performs the update step (:meth:`EKF.EKF.Update`) and logs the results.
        The method also plots the uncertainty ellipse (:meth:`PlotUncertainty`) of the robot pose, the feature observations and the expected feature observations.

        :param xk_1: previous state vector
        :param Pk_1: previous covariance matrix
        :return xk, Pk: updated state vector and covariance matrix
        """
        uk, Qk = self.GetInput()
        self.xk_1=xk_1
        self.Pk_1=Pk_1

        (self.xk_bar, self.Pk_bar) = self.Prediction(uk, Qk, self.xk_1, self.Pk_1) if uk.size > 0 else (self.xk_1, self.Pk_1)

        zm, Rm, Hm, Vm = self.GetMeasurements();
        zf, Rf = self.GetFeatures();

        self.H = self.DataAssociation(self.xk_bar, self.Pk_bar,  zf, Rf)
        zk, Rk, Hk, Vk, znp, Rnp = self.StackMeasurementsAndFeatures(zm, Rm, Hm, Vm,zf, Rf, self.H)
        (self.xk, self.Pk) = self.Update(zk, Rk, self.xk_bar, self.Pk_bar, Hk, Vk) if zk.size>0 else (self.xk_bar, self.Pk_bar)

        self.Log(self.robot.xsk, self.xk, self.Pk, self.xk_bar, zm)  # log the results for plotting

        self.PlotUncertainty(zf,Rf)

        return self.xk, self.Pk

    def StackMeasurementsAndFeatures(self, zm, Rm, Hm, Vm, zf, Rf, H):
        """
        Given the vector of  measurements observations :math:`z_m` together with their covariance matrix :math:`R_m`,
        the vector of feature observations :math:`z_f` together with their covariance matrix :math:`R_f`, The measurement observation matrix :math:`H_m`, the
        measurement observation noise matrix :math:`V_m` and the vector of feature associations :math:`H`, this method
        returns the joint observation vector :math:`z_k`, its related covariance matrix :math:`R_k`, the stacked
        Observation matrix :math:`H_k`, the stacked noise observation matrix :math:`V_k`, the vector of non-paired features
        :math:`z_{np}` and its noise covariance matrix :math:`R_{np}`.
        It is assumed that the measurements and the features observations are independent, therefore the covariance matrix
        of the joint observation vector is a block diagonal matrix.

        :param zm: measurement observations vector
        :param Rm: covariance matrix of the measurement observations
        :param Hm: measurement observation matrix
        :param Vm: measurement observation noise matrix
        :param zf: feature observations vector
        :param Rf: covariance matrix of the feature observations
        :param H: features associations vector
        :return: vector of joint measurement and feature observations :math:`z_k` and its covariance matrix :math:`R_k`
        """

        zp, Rp, Hp, Vp, znp, Rnp = self.SplitFeatures(zf, Rf, H)
        if zm.size == 0:
            zk, Rk, Hk, Vk = zp, Rp, Hp, Vp
        elif zp.size == 0:
            zk, Rk, Hk, Vk = zm, Rm, Hm, Vm
        else:
            zk, Rk, Hk, Vk = np.block([[zm], [zp]]), scipy.linalg.block_diag(Rm, Rp), np.block([[Hm], [Hp]]), scipy.linalg.block_diag(Vm, Vp)

        return zk, Rk, Hk, Vk, znp, Rnp

    def SplitFeatures(self, zf, Rf, H):
        """
        Given the vector of feature observations :math:`z_f` and their covariance matrix :math:`R_f`, and the vector of
        feature associations :math:`H`, this function returns the vector of paired feature observations :math:`z_p` together with
        its covariance matrix :math:`R_p`, and the vector of non-paired feature observations :math:`z_{np}` together with its covariance matrix :math:`R_{np}`.
        The paired observations will be used to update the filter, while the non-paired ones will be considered as outliers.
        In the case of SLAM, they become new feature candidates.

        :param zf: vector of feature observations
        :param Rf: covariance matrix of feature observations
        :param H: hypothesis of feature associations
        :return: vector of paired feature observations :math:`z_p`, covariance matrix of paired feature observations :math:`R_p`, vector of non-paired feature observations :math:`z_{np}`, covariance matrix of non-paired feature observations :math:`R_{np}`.
        """
        zf=BlockArray(zf,self.zfi_dim)
        Rf=BlockArray(Rf,self.zfi_dim)

        Hp = np.zeros((0, self.xk_bar.shape[0]))  # empty matrix
        Vp = np.zeros((0, 0))  # empty matrix

        nzf = len(H)  # number of features in the observation Hypothesis
        zp, Rp, znp, Rnp = np.zeros((0, 1)), np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((0, 0))
        for i in range(nzf):  # for each feature observation in the observation Hypothesis
            if H[i] is not None:  # if the feature is associated
                zp = np.block([[zp], [zf[[i]]]])
                Rp = scipy.linalg.block_diag(Rp, Rf[[i,i]]) if Rp is not None else Rf[[i,i]]
                Hp=np.block([[Hp],[self.Jhfjx(self.xk_bar, H[i])]])
                Vp=scipy.linalg.block_diag(Vp,np.eye(self.zfi_dim))
            else:
                znp = np.block([[znp], [zf[[i]]]])
                Rnp = scipy.linalg.block_diag(Rnp, Rf[[i,i]]) if Rnp.size > 0 else Rf[[i,i]]
        return zp, Rp, Hp, Vp, znp, Rnp

    def PlotFeatureObservationUncertainty(self, zf, Rf, color):  # plots the feature observation uncertainty ellipse
        """
        Plots the uncertainty ellipse of the feature observations. This method is called by :meth:`FEKFMBL.PlotUncertainty`.

        :param zf: vector of feature observations
        :param Rf: covariance matrix of the feature observations
        """

        zf=BlockArray(zf,self.zfi_dim)
        Rf=BlockArray(Rf,self.zfi_dim)

        if zf is not None:
            # Remove previous feature observation ellipses
            for i in range(len(self.plt_zf_ellipse)):
                self.plt_zf_ellipse[i].remove()
                self.plt_zf_line[i].remove()
            self.plt_zf_ellipse = []
            self.plt_zf_line = []

        NxB = self._GetRobotState(self.robot.xsk)

        # For all feature observations
        nzf = 0 if zf is None else zf.size // self.zfi_dim
        for i in range(0, nzf):
            BxF = self.Feature(zf[[i]])  # feature observation in the B-Frame
            BRF = Rf[[i,i]]  # feature observation covariance in the B-Frame
            NxF = self.g(NxB, BxF)
            J = self.Jgv(NxB, BxF)
            NRf = J @ BRF @ J.T
            NxF_Plot = NxF.ToCartesian()
            NRF_Plot = NxF.J_2c() @ NRf @ NxF.J_2c().T
            feature_ellipse = GetEllipse(NxF_Plot, NRF_Plot)
            plt_ellipse, = plt.plot(feature_ellipse[0], feature_ellipse[1], color)
            plt_line, = plt.plot([self.robot.xsk[0], NxF_Plot[0]],
                                 [self.robot.xsk[1], NxF_Plot[1]], color+'-.')
            self.plt_zf_ellipse.append(plt_ellipse)
            self.plt_zf_line.append(plt_line)
            #for j in range(len(self.M)): print("M[",j,"]=", self.M[j].ToCartesian().T)

    def PlotExpectedFeaturesObservationsUncertainty(self):
        """
        For all features in the map, this method plots the uncertainty ellipse of the expected feature observations. This method is called by :meth:`FEKFMBL.PlotUncertainty`.
        """
        for i in range(len(self.plt_hf_ellipse)):
            self.plt_hf_ellipse[i].remove()
        self.plt_hf_ellipse = []

        # Plot Expected Feature Observation Ellipses
        for Fj in range(self.nf):   # for all map features
            h_Fj = self.Feature(self.hfj(self.xk, Fj)) # expected feature observation in the B-Frame in the observation space
            J = self.Jhfjx(self.xk, Fj)
            P_h_Fj = J @ self.Pk @ J.T # expected feature observation covariance in the B-Frame in the observation space

            Nhx_Fj = self.g(self.xk, h_Fj) # expected feature observation in the N-Frame in storage space
            #Jx = self.Jgx(self.xk, h_Fj)
            Jv = self.Jgv(self.xk, h_Fj)

            #NP_Fj = Jx @ self.Pk @ Jx.T + Jv @ P_h_Fj @ Jv.T # expected feature observation covariance in the N-Frame in storage representation
            NP_Fj = Jv @ P_h_Fj @ Jv.T  # expected feature observation covariance in the N-Frame in storage representation

            ellipse = GetEllipse(Nhx_Fj.ToCartesian(), Nhx_Fj.J_2c() @ NP_Fj @ Nhx_Fj.J_2c().T)
            plt_ellipse, = plt.plot(ellipse[0], ellipse[1], 'black')  # plot it
            self.plt_hf_ellipse.append(plt_ellipse)  # and add it to the list

            #self.PlotSampleObservationSpace(self.xk, h_Fj, P_h_Fj, 100)

    def PlotSampleObservationSpace(self, NxB, BxFj, BPFj, n, color='r.'):
        """
        Plots n samples from a Gaussian distribution with mean :math:`x` and covariance :math:`P`. This method is called by :meth:`FEKFMBL.PlotUncertainty`.
        This is a method for testing. It can be used to compare the uncertainty ellipse with the samples.

        :param x: mean of the Gaussian distribution
        :param P: covariance of the Gaussian distribution
        :param n: number of samples
        :param color: color of the samples
        """
        ns = len(self.plt_samples)
        if ns > 0:
            for i in range(ns): self.plt_samples[i].remove()
            self.plt_samples = []

        Bsample = np.random.multivariate_normal(BxFj[0:,0],BPFj, n).T

        for i in range(Bsample.shape[1]):
            Nsample=self.g(NxB,self.Feature(Bsample[0:,i].reshape(2,1))).ToCartesian()
            plt_sample, = plt.plot(Nsample[0], Nsample[1], color, markersize=0.5)
            self.plt_samples.append(plt_sample)

    def PlotSample(self, x, P, n, color='r.'):
        """
        Plots n samples from a Gaussian distribution with mean :math:`x` and covariance :math:`P`. This method is called by :meth:`FEKFMBL.PlotUncertainty`.
        This is a method for testing. It can be used to compare the uncertainty ellipse with the samples.

        :param x: mean of the Gaussian distribution
        :param P: covariance of the Gaussian distribution
        :param n: number of samples
        :param color: color of the samples
        """
        ns = len(self.plt_samples)
        if ns > 0:
            for i in range(ns): self.plt_samples[i].remove()
            self.plt_samples = []

        sample = np.random.multivariate_normal(x[0:2, 0], P[0:2, 0:2], n).T

        for i in range(sample.shape[1]):
            plt_sample, = plt.plot(sample[0, i], sample[1, i], color, markersize=0.5)
            self.plt_samples.append(plt_sample)

    def PlotRobotUncertainty(self):  # plots the robot trajectory and its uncertainty ellipse
        """
        Plots the robot trajectory and its uncertainty ellipse. This method is called by :meth:`FEKFMBL.PlotUncertainty`.

        """
        # Plot Robot Ellipse
        robot_ellipse = GetEllipse(self.robot.xsk, self._GetRobotPoseCovariance(self.Pk))
        self.plt_robotEllipse.set_data(robot_ellipse[0], robot_ellipse[1])  # update it

        # Plot Robot Trajectory
        self.xTraj.append(self.xk[0, 0])
        self.yTraj.append(self.xk[1, 0])
        self.trajectory.pop(0).remove()
        self.trajectory = plt.plot(self.xTraj, self.yTraj, marker='.', color='blue', markersize=1)

    def PlotUncertainty(self,zf,Rf):
        """
        Plots the uncertainty ellipses of the robot pose (:meth:`PlotRobotUncertainty`), the feature observations
        (:meth:`PlotFeatureObservationUncertainty`) and the expected feature observations (:meth:`PlotExpectedFeaturesObservationsUncertainty`).
        This method is called by :meth:`FEKFMBL.Localize` at the end of a localization iteration in order to update
        the online  visualization.

        :param zf: vector of feature observations
        :param Rf: covariance matrix of the feature observations
        """
        if self.k % self.robot.visualizationInterval == 0:
            self.PlotRobotUncertainty()
            self.PlotFeatureObservationUncertainty(zf, Rf,'b')
            self.PlotExpectedFeaturesObservationsUncertainty()

    def _GetRobotState(self, xk):
        """
        Returns the robot state from the state vector.

        :param xk: mean of the state vector:math:`x_k`
        :return: mean robot state :math:`x_{B_k}`
        """
        #return xk[0:self.xB_dim]
        xk=BlockArray(xk,self.xF_dim, self.xB_dim, self.xBpose_dim)
        return xk[[FEKFMBL.xB]]

    def _SetRobotState(self, xk, xB):
        """
        Updates the robot state within the state vector.

        :param xk: mean of the state vector:math:`x_k`
        :param xB: mean robot state :math:`x_{B_k}`
        :return: updatd mean  state vector :math:`x_{k}`
        """
        #xk[0:self.xB_dim] = xB

        xk=BlockArray(xk,self.xF_dim, self.xB_dim, self.xBpose_dim)
        xk[[FEKFMBL.xB]]=xB

        return xk

    def _GetRobotStateCovariance(self, Pk):
        """
        Returns the robot covariance from the state covariance matrix.

        :param Pk: state vector covariance matrix :math:`P_k`
        :return: robot state covariance :math:`P_{B_k}`
        """
        return Pk[0:self.xB_dim, 0:self.xB_dim]

    def _SetRobotStateCovariance(self, Pk, PB):
        """
        Updates the robot covariance from the state covariance matrix.

        :param Pk: state vector covariance matrix :math:`P_k`
        :param PB: robot state covariance :math:`P_{B_k}`
        :return: updatd state covariance matrix :math:`P_{k}`
        """
        Pk[0:self.xB_dim, 0:self.xB_dim] = PB
        return Pk

    def _SetRobotPose(self, xk, xB):
        """
        Updates the robot pose within the state vector.

        :param xk: mean of the state vector:math:`x_k`
        :param xB: mean robot pose :math:`x_{B_k}`
        :return: updatd mean  state vector :math:`x_{k}`
        """
        xk[0:self.xB_dim] = xB
        return xk

    def _GetRobotPoseCovariance(self, Pk):
        """
        Returns the robot pose covariance from the state covariance matrix.

        :param Pk: state vector covariance matrix :math:`P_k`
        :return: robot pose covariance :math:`P_{B_k}`
        """
        return Pk[0:self.xBpose_dim, 0:self.xBpose_dim]

    def _SetRobotPoseCovariance(self, Pk, PB):
        """
        Updates the robot pose covariance from the state covariance matrix.

        :param Pk: state vector covariance matrix :math:`P_k`
        :param PB: robot pose covariance :math:`P_{B_k}`
        :return: updated state covariance matrix :math:`P_{k}`
        """
        Pk[0:self.xBpose_dim, 0:self.xBpose_dim] = PB
        return Pk

    # def FeatureObservation(self, zf, i):
    #     """
    #     Returns the i-th feature observation from the feature observation vector.
    #
    #     :param zf: vector of feature observations :math:`z_f`
    #     :param i: index of the feature observation
    #     :return: ith feature observation :math:`z_{f_i}`
    #     """
    #     v=BlockArray(zf,2)
    #
    #     return v[[i]] #GetBlockVector(zf,i,2)
    #
    #
    # def FeatureObservationCovariance(self, Rf, i):
    #     """
    #     Returns the i-th feature observation covariance from the feature observation covariance matrix.
    #
    #     :param Rf: feature observation covariance matrix :math:`R_f`
    #     :param i: index of the feature observation
    #     :return: covariance of the ith feature observation :math:`R_{f_i}`
    #     """
    #     Rfi = Rf[i * self.zfi_dim:(i + 1) * self.zfi_dim, i * self.zfi_dim:(i + 1) * self.zfi_dim].reshape(self.zfi_dim,self.zfi_dim)
    #     R=BlockArray(Rf,2)
    #     return R[[i,i]] #Rfi

# def GetBlockVector(v, i, dim):
#     return v[i * dim:(i + 1) *dim]
#
# def SetBlockVector(ov, iv, i, dim):
#     ov[i * dim:(i + 1) *dim]=iv
#
# def GetBlockDiagMatrix(M,i,dim):
#     BM=M[i * dim:(i + 1) *dim,i * dim:(i + 1) *dim] if M.size>0 else np.zeros((0,0))
#     return BM
#
# def SetBlockDiagMatrix(OM,IM,i,dim):
#     OM[i * dim:(i + 1) *dim,i * dim:(i + 1) *dim]=IM

