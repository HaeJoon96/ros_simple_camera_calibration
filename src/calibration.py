#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import copy
import math
import yaml
import numpy as np
from numpy.core.fromnumeric import put
from numpy.lib.arraysetops import isin
from moveit_commander.roscpp_initializer import roscpp_initialize

import rospy
import tf
import tf.transformations
import moveit_commander
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotTrajectory, RobotState, DisplayTrajectory

from robotiq_2f_gripper_control.robotiq_2f_gripper_ctrl import RobotiqCGripper
from simple_camera_calibration.srv import SimpleCameraCalibrationRequest, SimpleCameraCalibrationResponse, SimpleCameraCalibration
import geometry_msgs.msg
from std_msgs.msg import Int32
from std_msgs.msg import Bool

import pdb



class MotionPlanner(object):
    def __init__(self, group_name, pose_reference_frame):
        # get init variables
        self.group_name = group_name
        self.pose_reference_frame = pose_reference_frame

        # init tf
        self.tf = tf.TransformListener()
        rospy.sleep(1)  # wait for get /tf topic

        # init move group
        self.move_group = moveit_commander.MoveGroupCommander(group_name)
        self.move_group.set_pose_reference_frame(self.pose_reference_frame)

        # publishers
        self.display_planned_path_pub = rospy.Publisher(
            '/move_group/display_planned_path',
            DisplayTrajectory,
            queue_size=2)
        
        self.robotiq_2f_gripper = RobotiqCGripper()

    ##########
    #Gripper #
    ##########
        # get robotiq 2F gripper interface
        
        self.get_current_pose()

    def get_current_pose(self):
        """Wrapper for MoveGroupCommander.get_current_pose()

        Even the pose reference frame has been set, the MoveGroupCommander.get_current_pose()
        does not return a current pose correctly. Here we enforce to transform the
        current pose from the world frame to the reference frame.

        Returns:
            `geometry_msgs/PoseStamped`: Current end-effector pose on the pose reference frame.
        """
        return self.tf.transformPose(self.pose_reference_frame, self.move_group.get_current_pose())

    def now_position(self):
        print(self.get_current_pose())

    def activate_gripper(self):
        self.robotiq_2f_gripper.reset()
        rospy.sleep(0.01)
        self.robotiq_2f_gripper.wait_for_connection()
        if self.robotiq_2f_gripper.is_reset():
            self.robotiq_2f_gripper.reset()
            self.robotiq_2f_gripper.activate()
        self.robotiq_2f_gripper.goto(-1, 0.1, 30, block=True) # close
        self.robotiq_2f_gripper.goto(0.8, 0.1, 30, block=True) # close
    
    def get_target_plan(self, target_pose):
        self.move_group.set_pose_target(target_pose)
        plan = self.move_group.plan()
        return plan


    def get_cartesian_plan(self , target):
        waypoints = []
        waypoints.append(target.pose)
        (plan, fraction) = self.move_group.compute_cartesian_path(
                waypoints,
                eef_step=0.02,
                jump_threshold=0.0,
            )
        if len(plan.joint_trajectory.points) == 0:
    
            rospy.logwarn('[{}] Picking target pose motion plan failed!'.format(self.group_name))
            return None
        return plan



    def set_joint_plan(self,joint_angles):
        joint_goal = self.get_currnet_joint_values()
        for i , angle in enumerate (joint_angles):
            joint_goal[i] = np.deg2rad(angle)
        plan = self.move_group.plan(joint_goal)
        return plan

    def get_currnet_joint_values(self):
        return self.move_group.get_current_joint_values()

    def set_start_state_from_plan(self, plan):
        """Set start robot state as last robot state of the plan.
        Args:
            plan (`RobotTrajectory`): A motion plan.
        """
        assert isinstance(plan, RobotTrajectory)
    
        last_joint_state = JointState()
        last_joint_state.header = plan.joint_trajectory.header
        last_joint_state.name = plan.joint_trajectory.joint_names
        last_joint_state.position = plan.joint_trajectory.points[-1].positions
        moveit_robot_state = RobotState()
        moveit_robot_state.joint_state = last_joint_state
        self.move_group.set_start_state(moveit_robot_state)

    def set_start_state_to_current_state(self):
        self.move_group.set_start_state_to_current_state()

    def merge_plans(self, plans):
        """Merge multiple plans into one.
        Args:
            plans (`list` of `RobotTrajectory`): Plans to be merged.
        Returns:
            `RobotTrajectory`: A merged plan. If the plans are empty, return None.
        """
        # if plan is empty, return None
        if len(plans) == 0:
            rospy.logwarn('Plan list is empty.')
            return None

        # generate new empty plan
        merged_plan = RobotTrajectory()
        merged_plan.joint_trajectory.header = plans[0].joint_trajectory.header
        merged_plan.joint_trajectory.joint_names = plans[0].joint_trajectory.joint_names

        # append joint trajectory points
        last_time = rospy.Time()
        for plan in plans:
            # offset trajectory time
            for point in plan.joint_trajectory.points:
                point.time_from_start = rospy.Time(
                    last_time.secs + point.time_from_start.secs,
                    last_time.nsecs + point.time_from_start.nsecs)
            merged_plan.joint_trajectory.points += plan.joint_trajectory.points
            last_time = merged_plan.joint_trajectory.points[-1].time_from_start

        # remove duplicate points
        merged_plan = self.remove_duplicate_points_on_plan(merged_plan)
        return merged_plan

    def remove_duplicate_points_on_plan(self, plan):
        """Remove duplicate points on a plan.
        Args:
            plan (`RobotTrajectory`): A plan.
        Returns:
            `RobotTrajectory`: A plan with no duplicate points.
        """
        point_num = len(plan.joint_trajectory.points)
        for i in range(point_num-1):
            try:
                if plan.joint_trajectory.points[i].positions == plan.joint_trajectory.points[i+1].positions:
                    plan.joint_trajectory.points.pop(i)
            except IndexError:
                pass
        return plan

    def display_planned_path(self, plan):
        """Display planned path in RViz.
        Args:
            plan (RobotTrajectory): A motion plan to be displayed.
        """   
        # assert type
        assert isinstance(plan, (list, RobotTrajectory))

        # generate msg
        display_trajectory = DisplayTrajectory()
        display_trajectory.model_id = self.group_name
        display_trajectory.trajectory_start = self.move_group.get_current_state()

        # check the type and put correct trajectory
        if isinstance(plan, list):
            for p in plan:
                assert isinstance(p, RobotTrajectory)
            display_trajectory.trajectory = plan
        elif isinstance(plan, RobotTrajectory):
            display_trajectory.trajectory.append(plan)

        # publish the msg
        self.display_planned_path_pub.publish(display_trajectory)

    def execute_plan_auto(self, plan, auto=False):
        """
        A wrapper for MoveIt function move_group.execute(). Makes the robot do the executed the given plan.
        --------------
        plan : RobotTrajectory() (MoveIt RobotTrajectory MSG)
            A motion plan to be executed
        auto : `Bool`
            If true, robot will be executed with out user confirmation.
        --------------
        """
        if plan == None:
            raise ValueError("Plan is non existant.")
        
        self.display_planned_path(plan)

        if auto is True:
            # raise NotImplementedError('Auto execution is no implemented')
            self.move_group.execute(plan, wait=True)
            self.move_group.stop()
        elif auto is False:
            # rospy.loginfo("Excute the plan")
            self.confirm_to_execution()
            self.move_group.execute(plan, wait=True)
            self.move_group.stop()
            rospy.loginfo("Finish execute")
        plan = None


    def confirm_to_execution(self):
        while True:
            ask_for_execution = '[{}] Robot execution is requested. Excute? (y/n)'.format(self.group_name)
            rospy.logwarn(ask_for_execution)
            cmd = raw_input()

            if (cmd == 'y') or (cmd == 'Y'):
                rospy.logwarn('[{}] Got positive responce. Excute planned motion!'.format(self.group_name))
                break
            elif (cmd == 'n') or (cmd == 'N'):
                info_str = '[{}] Got ABORT signal. Shutdown!!'.format(self.group_name)
                rospy.logerr(info_str)
                raise Exception(info_str)
            else:
                rospy.logwarn('[{}] Command should be `y` or `n`. Got `{}`'.format(self.group_name, cmd))

class Calibraiont_Planner(MotionPlanner):
    def __init__(self, motion_planner):
        assert isinstance(motion_planner, MotionPlanner)
        self.motion_planner = motion_planner
        self.camear_calibraiton = rospy.ServiceProxy('camera_calibration', SimpleCameraCalibration)

    def Calibration_plan(self):
        calib_pose = self.set_calib_pose()
        calib_plan =  self.motion_planner.set_joint_plan(calib_pose)
        self.motion_planner.execute_plan_auto(calib_plan, auto= True)
        self.cal_center_pose()
    
    def set_calib_pose(self):
        home_pose_joint_angle = [90,-75,30,0,45,0]
        return home_pose_joint_angle

    def cal_center_pose(self):
        current_pose = self.motion_planner.get_current_pose()
        pose = copy.deepcopy(current_pose)
        center_pose = self.motion_planner.tf.transformPose(self.motion_planner.pose_reference_frame, pose)
        req = SimpleCameraCalibrationRequest()
        req.hand_base_tf.transform.translation = center_pose.pose.position
        req.hand_base_tf.transform.rotation = center_pose.pose.orientation
        self.camear_calibraiton(req)


if __name__ == '__main__':
    rospy.init_node('test')
    auto_execute = True

    irb120_reference_frame = 'irb120_base'
    calib_motion_planner1 = MotionPlanner(group_name='irb120_cal', pose_reference_frame=irb120_reference_frame)
 
    calib_motion_planner = Calibraiont_Planner(calib_motion_planner1)
    calib_motion_planner.Calibration_plan()
    

