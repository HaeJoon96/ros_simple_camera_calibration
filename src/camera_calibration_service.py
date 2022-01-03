#!/usr/bin/env python2

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, TransformStamped
from simple_camera_calibration.srv import SimpleCameraCalibration, SimpleCameraCalibrationResponse
import tf2_ros as tf2
from tf import transformations
import utils


class CameraCalibrationService(object):
    def __init__(self, pattern_size):
        self.pattern_size = pattern_size

        self.bridge = CvBridge()
        self.objpoints = []
        self.imgpoints = []

        self.camera_calibration_service = rospy.Service('camera_calibration', SimpleCameraCalibration, self.camera_calibration_callback)

        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.img_cb)
        self.camera_info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.info_cb)

        self.tf_broadcaster = tf2.StaticTransformBroadcaster()

    def img_cb(self, img):
        try:
            self.color_msg = img
            self.color_img = self.bridge.imgmsg_to_cv2(self.color_msg,"bgr8")
        except CvBridgeError as e:
            print(e)
        return

    def info_cb(self, msg):
        """
        Callback for the camera information.
        ----------
        Args:
            msg {CameraInfo}: The camera information message.
        ----------
            self.K {numpy.array}: The camera matrix.
            self.D {numpy.array}: The distortion coefficients.
        """
        self.K = np.reshape(msg.K,(3,3))    # Camera matrix
        self.D = np.array(msg.D) # Distortion matrix. 5 for IntelRealsense, 8 for AzureKinect

    def camera_calibration_callback(self, req):
        
        response = SimpleCameraCalibrationResponse()
        assert (isinstance(response, SimpleCameraCalibrationResponse))
        # Get the image
        image = self.color_img
        #image = cv2.undistort(image, self.K, self.D)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        # Make the hand base transfrom into a matrix
        hand_base_tf = req.hand_base_tf
        hand_base_tf_trans = np.array([hand_base_tf.transform.translation.x, hand_base_tf.transform.translation.y, hand_base_tf.transform.translation.z])
        hand_base_tf_rot = np.array([hand_base_tf.transform.rotation.x, hand_base_tf.transform.rotation.y, hand_base_tf.transform.rotation.z, hand_base_tf.transform.rotation.w])
        hand_base_tf_mat = utils.quat_trans_to_matrix(hand_base_tf_trans, hand_base_tf_rot)

        # Find the cheesboard corners
        objp = np.zeros((1, self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        x = np.linspace(50, -50, 6, endpoint=True) / 1000
        y = np.linspace(-40, 40, 5, endpoint=True) / 1000
        xv, yv = np.meshgrid(x, y, indexing='xy')
        for i in range(objp.shape[1]):
            x_ind = i // self.pattern_size[0]
            y_ind = i % self.pattern_size[0]
            objp[0,i,:] =np.array([xv[x_ind, y_ind], yv[x_ind, y_ind], 0])

        
        #print(gray)
        retval, corners = cv2.findChessboardCorners(gray, self.pattern_size, None) #cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        # Optimize the corners
        if retval == True:
            print("Found corners")
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            cv2.drawChessboardCorners(image, self.pattern_size, corners_2, retval)
            # self.temp_pub = rospy.Publisher("/chesboard_img", Image, 2)
            # out_img = Image()
            # out_img = self.bridge.cv2_to_imgmsg(image, "bgr8")
            # for i in range(100):
            #     self.temp_pub.publish(out_img)
            #     rospy.sleep(0.1)
            #cv2.imshow('img', gray)
            #cv2.waitKey(500)

            self.objpoints.append(objp)
            self.imgpoints.append(corners_2)
        else:
            print("Corners not found")
            response.success.data = False
            return response

        # Calibrate the camera
        # retval, _, _, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape, self.K, self.D)
        flags = cv2.SOLVEPNP_ITERATIVE
        # solve pnp
        (retval, rvecs, tvecs) = cv2.solvePnP(
            self.objpoints[0],
            self.imgpoints[0],
            self.K,
            self.D,
            flags=flags)

        hand_camera_tf = self.make_pose(rvecs, tvecs)
        hand_camera_tf_mat = utils.pose_to_matrix(hand_camera_tf)
        camera_hand_tf_mat = transformations.inverse_matrix(hand_camera_tf_mat)

        camera_base_tf_mat = np.dot(hand_base_tf_mat, camera_hand_tf_mat)
        rospy.logwarn("Camera base tf: {}".format(camera_base_tf_mat))
        camera_base_tf = utils.matrix_to_pose(camera_base_tf_mat)

        # Make the camera calibration response
        response.success.data = True
        response.camera_base_tf.transform.translation = camera_base_tf.position
        response.camera_base_tf.transform.rotation = camera_base_tf.orientation
        response.camera_base_tf.header.frame_id = "irb120_base"
        response.camera_base_tf.child_frame_id = "camera_color_frame"
        response.camera_base_tf.header.stamp = rospy.Time.now()
        self.tf_broadcaster.sendTransform(response.camera_base_tf)

        return response


    def make_pose(self, rvec, tvec):
        """
        Given a rotation vector and a translation vector, returns a cheesboard Pose.
        ----------
        Args:
            id {int} -- id of the marker
            rvec {np.array} -- rotation vector of the marker
            tvec {np.array} -- translation vector of the marker
        ----------
        Returns:
            Pose -- Pose of the marker
        """
        marker_pose = Pose()
        tvec = np.squeeze(tvec)
        rvec = np.squeeze(rvec)

        r_mat = np.eye(3)
        cv2.Rodrigues(rvec, r_mat)
        tf_mat = np.eye(4)
        tf_mat[0:3,0:3] = r_mat

        quat = transformations.quaternion_from_matrix(tf_mat)

        marker_pose.position.x = tvec[0]
        marker_pose.position.y = tvec[1]
        marker_pose.position.z = tvec[2]

        marker_pose.orientation.x = quat[0]
        marker_pose.orientation.y = quat[1]
        marker_pose.orientation.z = quat[2]
        marker_pose.orientation.w = quat[3]

        return marker_pose


if __name__ == '__main__':
    rospy.init_node('camera_calibration_service')

    camera_calibration_service = CameraCalibrationService((6,5))

    rospy.spin()