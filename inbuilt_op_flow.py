#!/usr/bin/env python

from std_msgs.msg import Float64
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Vector3Stamped, Twist
# from gazebo_msgs.msg import ModelStates
import numpy as np
from scipy import signal, ndimage
from cv_bridge import CvBridge, CvBridgeError
import cv2
# from PIL import Image as IM
# import keyboard
import rospy
import time
from sys import float_info
import math
# from imutils import rotate




class op_flow():  
    def __init__(self):

        rospy.init_node('op_flow') #Intializing the solver node

#initialisations:

        # params for ShiTomasi corner detection
        self.feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15, 15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        self.color = np.random.randint(0, 255, (100, 3))
        self.mask = np.uint8(np.zeros([100,120]))
        

        #Frequency of operation.
        self.sample_rate = 10.0 #10 Hz (better to keep it around 50)
        
        #for publising the velocity command
        self.vel_cmd = Twist()
        self.vel_cmd.linear.x = 0
        self.vel_cmd.linear.y = 0
        self.vel_cmd.linear.z = 0
        self.vel_cmd.angular.x = 0
        self.vel_cmd.angular.y = 0
        self.vel_cmd.angular.z = 0

        self.img = np.empty([])
        self.img_prev = np.empty([])
        self.time_stamp1 = 0
        self.time_stamp2 = 0
        self.bridge = CvBridge()

        self.ww = 10
        self.w = int(round(self.ww/2))
        self.step = 10
        self.count = 0
        self.ad = '/home/aman/output/'
        self.img_depth = np.zeros([1,1])
        self.flag = 0
        self.local_flag = 0
        self.t1 = float_info.max
        self.gauss = np.array([[]])
        self.theta = 0

        ##Gaussian_____________________________________________________________________________
        kernel_size = 2*self.w+1
        sigma, mu = 0.5, 0.0

        x, y = np.meshgrid(np.linspace(-1,1,kernel_size), np.linspace(-1,1,kernel_size))
        d = np.sqrt(x*x+y*y)
        
        self.gauss = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )


        #angular velocity from imu
        self.omega_x = 0
        self.omega_y = 0
        self.omega_z = 0

        #linear accelaration from imu
        self.lin_acc_x = 0
        self.lin_acc_y = 0
        self.lin_acc_z = 0

        #linear velocity from imu
        self.lin_vel_x = 0
        self.lin_vel_y = 0
        self.lin_vel_z = 0

        #translational vel from gps (ground truth)
        self.velgt_x = 0
        self.velgt_y = 0
        self.velgt_z = 0

        #translational vel from camera
        self.vel_x = 0
        self.vel_y = 0
        self.vel_z = 0

        #constants for velocity calculation from image
        self.focal = 10
        self.velx_avg = np.array([0])
        self.vely_avg = np.array([0])

        #butterworth filter for camera_____________________

        N_cam = 3  #filter order
        fs_sig_cam = 10 #sampling freq (Hz)
        Wn_cam = 0.85 #cutoff freq (Hz)
        

        self.b_cam, self.a_cam = signal.butter(N_cam, Wn_cam, btype='lp', analog=False, output='ba', fs=fs_sig_cam)
        print("coeff = ", self.b_cam, self.a_cam)
        self.b_cam = np.array([self.b_cam, self.b_cam])
        self.a_cam = np.array([self.a_cam, self.a_cam])

        self.x_cam = np.zeros(np.shape(self.b_cam))
        self.y_cam = np.zeros(np.shape(self.a_cam))

        #butterworth filter for imu________________________

        N_imu = 1 #filter order
        fs_sig_imu = 100 #sampling freq (Hz)
        Wn_imu = 0.05 #cutoff freq (Hz)
        

        self.b_imu, self.a_imu = signal.butter(N_imu, Wn_imu, btype='hp', analog=False, output='ba', fs=fs_sig_imu)
        # self.b_imu, self.a_imu = signal.cheby1(N_imu,0.5, Wn_imu,btype='high',analog=False,output='ba', fs=fs_sig_imu)
        print("coeff = ", self.b_imu, self.a_imu)
        self.b_imu = np.array([self.b_imu, self.b_imu])
        self.a_imu = np.array([self.a_imu, self.a_imu])

        self.x_imu = np.zeros(np.shape(self.b_imu))
        self.y_imu = np.zeros(np.shape(self.a_imu))
        

        #extra variable for plotting
        self.plotvx = np.array([0])
        self.plotvxgt = np.array([0])
        self.plotvy = np.array([0])
        self.plotvygt = np.array([0])
        self.plotang = np.array([0])
        self.plot_lin_vx = np.array([0])
        self.plot_lin_vy = np.array([0])
        self.plot_lin_vx_final = np.array([0])
        self.plot_lin_vy_final = np.array([0])
        

        #accumulator for 5 point moving average MA filter (higher lenght changes phase more)
        # self.acc_x = np.zeros([5,1])
        # self.acc_y = np.zeros([5,1])

        #publishers
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size= 1)
   
        
        #Subscribers
        rospy.Subscriber('/cam_ir/color/image_raw', Image, self.image_callback)
        rospy.Subscriber('/cam_ir/depth/image_raw', Image, self.depth_image_callback)
        rospy.Subscriber('/imu', Imu, self.imu_callback)
        rospy.Subscriber('/gps/fix_velocity', Vector3Stamped, self.gps_callback)
        # rospy.Subscriber('/gazebo/model_states ', Vector3Stamped, self.gazebo_callback)
        


#callbacks

  
    def image_callback(self, msg):
        try:
        
            # if self.flag == 0 and self.local_flag == 0:  #ensure processing is not happening and it is the first frame
            #     self.t1 = time.time()
            #     self.img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            #     self.local_flag = 1                     #deny entry till next consecutive frame gets loaded

            # if (time.time() - self.t1) > 1/20 and self.flag == 0:    #fps of 20 for the two frames
            #     print(time.time() - self.t1)
            #     self.img_prev = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            #     self.flag = 1
            #     self.local_flag = 0  
            #     print('obtained image')  
            
            # if self.flag == 0 and self.local_flag == 0:  #ensure processing is not happening and it is the first frame
            #     self.img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            #     print(self.local_flag)

            # if self.flag == 0 and self.local_flag == 1:
            #     self.img_prev = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            #     self.flag = 1
            #     self.local_flag = 0
            #     return

            # self.local_flag = 1

            if self.local_flag == 0 and self.flag == 0:
                self.img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.local_flag = 1
                # print("got img")
                self.time_stamp1 = time.time()

            elif self.flag == 0:
                # time.sleep(0.1)
                self.img_prev = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.local_flag = 0
                self.flag = 1
                # print("got prev_img")
                self.time_stamp2 = time.time()

            
        except CvBridgeError as e:
            print(e)
            return
        # cv2.imshow('image', self.img)
        # cv2.waitKey(5)

    def depth_image_callback(self, msg):
        try:
            #convert ros msg to cv2 copatible image and recoprocate (as inf causes normalization error)
            self.img_depth = np.reciprocal(self.bridge.imgmsg_to_cv2(msg, "32FC1"))
            # cv2.imshow('deth_map',self.img_depth)
            # cv2.waitKey(5)
            
        except CvBridgeError as e:
            print(e)
            return
        # cv2.imshow('image', self.img)
        # cv2.waitKey(5)

    def imu_callback(self,msg):
        self.omega_x = msg.angular_velocity.x
        self.omega_y = msg.angular_velocity.y
        self.omega_z = msg.angular_velocity.z

        self.lin_acc_x = msg.linear_acceleration.x + (np.random.rand()-0.45)
        self.lin_acc_y = msg.linear_acceleration.y + (np.random.rand()-0.45)
        self.lin_acc_z = msg.linear_acceleration.z + (np.random.rand()-0.45)

        self.lin_vel_x += self.lin_acc_y/10 # x and y are swapped
        self.lin_vel_y -= self.lin_acc_x/10
        self.lin_vel_z += (self.lin_acc_z+9.8)/10



        # print(self.omega_x, self.omega_y, self.omega_z)

    def gps_callback(self, msg):
        self.velgt_x = msg.vector.x
        self.velgt_y = msg.vector.y
        self.velgt_z = msg.vector.z

        # print(self.velgt_x, self.velgt_y, self.velgt_z)

    # def gazebo_callback(self, msg):
    #     print(msg)

    ## butterworth filter for camera
    def butterworth_filt_cam(self, sig):

        self.x_cam[:,1:] = self.x_cam[:,0:-1]

        self.x_cam[0,0] = sig[0]
        self.x_cam[1,0] = sig[1]

        temp1 = np.array([np.sum(self.x_cam[0][:]*self.b_cam[0][:]) - np.sum(self.a_cam[0][1:]*self.y_cam[0][0:-1])])
        temp2 = np.array([np.sum(self.x_cam[1][:]*self.b_cam[1][:]) - np.sum(self.a_cam[1][1:]*self.y_cam[1][0:-1])])

        for i in range(len(self.a_cam[0])-1):
            # print(temp)
            temp1 = np.append(temp1,self.y_cam[0][i])
            temp2 = np.append(temp2,self.y_cam[1][i])
            

        self.y_cam = [temp1,temp2]

        return [self.y_cam[0][0], self.y_cam[1][0]]

    ## butterworth filter for imu
    def butterworth_filt_imu(self, sig):

        self.x_imu[:,1:] = self.x_imu[:,0:-1]

        self.x_imu[0,0] = sig[0]
        self.x_imu[1,0] = sig[1]

        temp1 = np.array([np.sum(self.x_imu[0][:]*self.b_imu[0][:]) - np.sum(self.a_imu[0][1:]*self.y_imu[0][0:-1])])
        temp2 = np.array([np.sum(self.x_imu[1][:]*self.b_imu[1][:]) - np.sum(self.a_imu[1][1:]*self.y_imu[1][0:-1])])

        for i in range(len(self.a_imu[0])-1):
            # print(temp)
            temp1 = np.append(temp1,self.y_imu[0][i])
            temp2 = np.append(temp2,self.y_imu[1][i])
            

        self.y_imu = [temp1,temp2]

        return [self.y_imu[0][0], self.y_imu[1][0]]




        

#algorithm

    def solving_algo(self):  

        #optical flow
        if len(self.img.shape) > 0 and self.flag == 1: # assert that images are recieved (in correct order)

            # Frame obtain
            fr2 = self.img
            fr1 = self.img_prev

            im1 = cv2.cvtColor(fr1,cv2.COLOR_BGR2GRAY)
            # im1 = cv2.normalize(im1t.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            # im1 = ndimage.gaussian_filter(im1, sigma = 5)

            im2 = cv2.cvtColor(fr2,cv2.COLOR_BGR2GRAY)
            # im2 = cv2.normalize(im2t.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) 
            # im2 = ndimage.gaussian_filter(im2, sigma = 5)

            # Take first frame and find corners in it
            p0 = cv2.goodFeaturesToTrack(im1, mask = None, **self.feature_params)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(im1, im2, p0, None, **self.lk_params)
            # Select good points
            if p1 is not None:
                good_new = p1[st==1]
                good_old = p0[st==1]
            # # draw the tracks
            # self.mask = np.uint8(np.zeros([100,120]))
            # for i, (new, old) in enumerate(zip(good_new, good_old)):
            #     a, b = new.ravel()
            #     c, d = old.ravel()
            #     self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
            #     im1 = cv2.circle(im1, (int(a), int(b)), 5, self.color[i].tolist(), -1)
            # # print(type(frame[0,0]),type(self.mask[0,0]))
            # img = cv2.add(im1, self.mask)
            # cv2.imshow('frame', img)
            # k = cv2.waitKey(30) & 0xff
            # if k == 27:
            #     cv2.destroyAllWindows()


            
            # self.velx_avg = self.velx_avg[1:]
            # self.vely_avg = self.vely_avg[1:]
            # print(p0, np.reshape(p0,[p0.shape[0],p0.shape[2]]))
            point_1 = np.reshape(p0,[p0.shape[0],p0.shape[2]])
            point_2 = np.reshape(p1,[p1.shape[0],p1.shape[2]])
            self.vel_x = np.mean(point_2[:][1]-point_1[:][1])
            self.vel_y = np.mean(point_2[:][0]-point_1[:][0])
            

            # self.vel_x = (-nu[0]*math.cos(self.theta) + nu[1]*math.sin(self.theta))/self.focal
            # self.vel_y = (-nu[0]*math.sin(self.theta) - nu[1]*math.cos(self.theta))/self.focal

            # self.theta+=self.omega_z/(10)

            # self.vel_x = -nu[0]/self.focal
            # self.vel_y = -nu[1]/self.focal
            # self.vel_x = 0
            # self.vel_y = 0

            # self.vel_x, self.vel_y = self.butterworth_filt_cam([self.vel_x, self.vel_y])
            # self.lin_vel_x, self.lin_vel_y = self.butterworth_filt_imu([self.lin_vel_x, self.lin_vel_y])
            

            #correcting for rotational error
            # self.theta_optical_z = math.atan2(self.vel_y,self.vel_x)
            # self.omega_optical_z = (self.prev_theta_optical_z-self.theta_optical_z)*self.scaling #considering constant discrete time steps
 
            #storing in array for plotting
            # print(self.vel_x, self.velgt_x)
            self.plotvx = np.append(self.plotvx, self.vel_x)
            self.plotvy = np.append(self.plotvy, self.vel_y)
            self.plotvxgt = np.append(self.plotvxgt, self.velgt_y)
            self.plotvygt = np.append(self.plotvygt, self.velgt_x)
            # self.plot_lin_vx = np.append(self.plot_lin_vx, self.lin_vel_x)
            # self.plot_lin_vy = np.append(self.plot_lin_vy, self.lin_vel_y)
            # self.plot_lin_vx_final = np.append(self.plot_lin_vx_final, 0.5*(self.lin_vel_x+self.vel_x))
            # self.plot_lin_vy_final = np.append(self.plot_lin_vy_final, 0.5*(self.lin_vel_y+self.vel_y))
            # self.plotang = np.append(self.plotang, self.omega_optical_z)

            #restting
            self.velx_avg = [0]
            self.vely_avg = [0]
            self.acc_x = [0]
            self.acc_y = [0]

            # print(self.vel_x,self.velgt_x,"comparision")

            # np.savetxt(self.ad+str(self.count)+'u.csv', u, delimiter=',')
            # np.savetxt(self.ad+str(self.count)+'v.csv', v, delimiter=',')
            # # # np.savetxt(self.ad+str(self.count)+'depth.csv', self.img_depth, delimiter=',')
            # cv2.imwrite(self.ad+str(self.count)+'.png',fr2)
            
            print(self.count)
            
            if self.count == 200:
                np.savetxt(self.ad+'velx.csv', self.plotvx, delimiter=',')
                np.savetxt(self.ad+'velgtx.csv', self.plotvygt, delimiter=',')
                np.savetxt(self.ad+'vely.csv', self.plotvy, delimiter=',')
                np.savetxt(self.ad+'velgty.csv', self.plotvxgt, delimiter=',')
            #     np.savetxt(self.ad+'linvelx.csv', self.plot_lin_vx, delimiter=',')
            #     np.savetxt(self.ad+'linvely.csv', self.plot_lin_vy, delimiter=',')
            #     np.savetxt(self.ad+'linvelxfinal.csv', self.plot_lin_vx_final, delimiter=',')
            #     np.savetxt(self.ad+'linvelyfinal.csv', self.plot_lin_vy_final, delimiter=',')
            #     # np.savetxt(self.ad+'velang.csv', self.plotang, delimiter=',')
                print("done!!!")

            # end = time.time()
            self.count = self.count+1

            
            # if self.count < 100:
            #     # self.vel_cmd.linear.x = -math.sin((2*math.pi*10)*self.count/500)
            #     # self.vel_cmd.linear.y = -math.cos((2*math.pi*10)*self.count/500)
            #     self.vel_cmd.angular.z = 0.5
            #     self.vel_cmd.linear.x = -0.5
            #     self.vel_cmd.linear.y= 1

            # else:
            #     # self.vel_cmd.linear.x = -math.cos((2*math.pi*10)*self.count/500)
            #     # self.vel_cmd.linear.y = -math.sin((2*math.pi*10)*self.count/500)
            #     self.vel_cmd.angular.z = 0.5
            #     self.vel_cmd.linear.x = -0.5
            #     self.vel_cmd.linear.y= 0
            # self.vel_cmd.linear.x = -0.25
            # self.vel_cmd.angular.z = 0.5
            self.vel_cmd.linear.x = -math.sin((2*math.pi*10)*self.count/500)
            self.vel_cmd.linear.y = -math.cos((2*math.pi*10)*self.count/500)
            self.vel_pub.publish(self.vel_cmd)

            # print(self.velgt_x,'X',self.velgt_y,'y')

            # self.prev_theta_optical_z = self.theta_optical_z
            self.flag = 0   #inform that processing is done



if __name__ == '__main__':

    solver = op_flow() #Creating solver object
    r = rospy.Rate(solver.sample_rate) 
    while not rospy.is_shutdown():

        try:
            solver.solving_algo()
            r.sleep()
        except rospy.exceptions.ROSTimeMovedBackwardsException:
            pass