import math
import numpy as np
from scipy import linalg
import RPi.GPIO as GPIO
from time import sleep

# #pin definitions (board)
# M1_pwm_pin = 12
# M2_pwm_pin = 31
# M3_pwm_pin = 33
# M4_pwm_pin = 35
# 
# M1_dir_pin = 29
# M2_dir_pin = 16
# M3_dir_pin = 31
# M4_dir_pin = 18

#pin definitions (BCM)
M1_pwm_pin = 18
M2_pwm_pin = 12
M3_pwm_pin = 13
M4_pwm_pin = 19

M1_dir_pin = 24
M2_dir_pin = 6
M3_dir_pin = 5
M4_dir_pin = 23

#GPIO and PWM init
GPIO.setwarnings(False)			#disable warnings
GPIO.setmode(GPIO.BCM)		#set pin numbering system

GPIO.setup(M1_pwm_pin,GPIO.OUT)
GPIO.setup(M2_pwm_pin,GPIO.OUT)
GPIO.setup(M3_pwm_pin,GPIO.OUT)
GPIO.setup(M4_pwm_pin,GPIO.OUT)
GPIO.setup(M1_dir_pin,GPIO.OUT)
GPIO.setup(M2_dir_pin,GPIO.OUT)
GPIO.setup(M3_dir_pin,GPIO.OUT)
GPIO.setup(M4_dir_pin,GPIO.OUT)

#create PWM instance with frequency
M1_pwm = GPIO.PWM(M1_pwm_pin,500)		
M1_pwm.start(0)
M2_pwm = GPIO.PWM(M2_pwm_pin,500)		
M2_pwm.start(0)
M3_pwm = GPIO.PWM(M3_pwm_pin,500)		
M3_pwm.start(0)
M4_pwm = GPIO.PWM(M4_pwm_pin,500)		
M4_pwm.start(0)

def mec_kinematics(vx,vy,w):

    #vx = -2
    #vy = 0
    #w = 0.5

    V = np.array([[vx],[vy],[w]])

    #body measurements
    lx = 34/2/100  #in meters
    ly = 27.5/2/100
    r = 4.5/100  #wheel rad (m)

    theta = math.pi/4

    L = ly*math.tan(theta)-lx

    t_mat = [[-math.tan(theta), math.tan(theta), -math.tan(theta), math.tan(theta)],
             [1, 1, 1, 1],
             [L, L, -L, -L]]
    
    w_wheels = 1/r*np.matmul(linalg.pinv(t_mat),V)
    
    w_wheels = (np.round(w_wheels))
    return w_wheels

def set_controls(vx,vy,w,amp_factor):
    
    w_wheels = mec_kinematics(vx,vy,w)
    
    dir_val = np.zeros(4,dtype = int)
    w_wheels_abs = abs(w_wheels)
    
    
    for i in range(4):
        dir_val[i] = int(w_wheels[i]/w_wheels_abs[i] == 1)#may encounter div by zero, but not lethal
    print(dir_val)
    w_wheels_abs +=amp_factor
    print(w_wheels_abs)
        
    M1_pwm.ChangeDutyCycle(int(w_wheels_abs[0]))
    M2_pwm.ChangeDutyCycle(int(w_wheels_abs[1]))
    M3_pwm.ChangeDutyCycle(int(w_wheels_abs[2]))
    M4_pwm.ChangeDutyCycle(int(w_wheels_abs[3]))
    
    GPIO.output(M1_dir_pin, int(dir_val[0]))
    GPIO.output(M2_dir_pin, int(dir_val[1]))
    GPIO.output(M3_dir_pin, int(dir_val[2]))
    GPIO.output(M4_dir_pin, int(dir_val[3]))
    
if __name__ == "__main__":
    
    cont = 1 #continue =1, stop = 0
    count = 0
    while cont == 1:
        
        
        #take user inputs
        data_in = np.zeros(4)
        for i in range(4):
            data_in[i] = float(input('Enter data '+str(i+1))+'\n')
            
        vx = data_in[0]
        vy = data_in[1]                
        w = data_in[2]/20
        cont = data_in[3]

        # vx = 2*math.sin(count/math.pi*0.02)
        # vy =2*math.cos(count/math.pi*0.02)
        # w = 0
#         cont = 1
        
        #print(vx, vy, w, cont)
        amp_factor = 20 #bias to over come low pwm values
        
        set_controls(vx,vy,w,amp_factor)
        count = count+1
        sleep(0.02)
        
        
        










