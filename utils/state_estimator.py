import numpy as np
import struct


class MahonyIMU:
    def __init__(self):
        self.gravX = 0.0
        self.gravY = 0.0
        self.gravZ = 1.0
        self.baseZacc = 1.0

        # Use MAHONY Quaternion IMU
        self.two_kp = 2.0 * 0.4
        self.two_ki = 2.0 * 0.001

        self.integralFBx = 0.0
        self.integralFBy = 0.0
        self.integralFBz = 0.0

        self.qw = 1.0
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0

        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

    def sensfusion6UpdateQ(self, gx, gy, gz, ax, ay, az, dt):
        self.sensfusion6UpdateQImpl(gx, gy, gz, ax, ay, az, dt)
        self.estimatedGravityDirection()

    def sensfusion6UpdateQImpl(self, gx, gy, gz, ax, ay, az, dt):
        gx = gx * np.pi / 180
        gy = gy * np.pi / 180
        gz = gz * np.pi / 180

        if (ax != 0.0) and (ay != 0) and (az != 0):
            recip_norm = inv_sqrt(ax * ax + ay * ay + az * az)
            ax *= recip_norm
            ay *= recip_norm
            az *= recip_norm

            halfvx = self.qx * self.qz - self.qw * self.qy
            halfvy = self.qw * self.qx + self.qy * self.qz
            halfvz = self.qw * self.qw - 0.5 + self.qz * self.qz

            # Error is sum of cross product between estimated and measured direction of gravity
            halfex = ay * halfvz - az * halfvy
            halfey = az * halfvx - ax * halfvz
            halfez = ax * halfvy - ay * halfvx

            if self.two_ki > 0:
                self.integralFBx += self.two_ki * halfex * dt  # integral error scaled by Ki
                self.integralFBy += self.two_ki * halfey * dt
                self.integralFBz += self.two_ki * halfez * dt
                gx += self.integralFBx  # apply integral feedback
                gy += self.integralFBy
                gz += self.integralFBz

            else:
                self.integralFBx = 0.0
                self.integralFBy = 0.0
                self.integralFBz = 0.0

            # Apply proportional feedback
            gx += self.two_kp * halfex
            gy += self.two_kp * halfey
            gz += self.two_kp * halfez

        # Integrate rate of change of quaternion
        gx *= 0.5 * dt  #  pre-multiply common factors
        gy *= 0.5 * dt
        gz *= 0.5 * dt
        qa = self.qw
        qb = self.qx
        qc = self.qy
        self.qw += -qb * gx - qc * gy - self.qz * gz
        self.qx += qa * gx + qc * gz - self.qz * gy
        self.qy += qa * gy - qb * gz + self.qz * gx
        self.qz += qa * gz + qb * gy - qc * gx

        # Normalise quaternion
        recipNorm = inv_sqrt(self.qw * self.qw + self.qx * self.qx + self.qy * self.qy + self.qz * self.qz)
        self.qw *= recipNorm
        self.qx *= recipNorm
        self.qy *= recipNorm
        self.qz *= recipNorm

        return self.qx, self.qy, self.qz, self.qw
    
    def sensfusion6GetAccZ(self, ax, ay, az):
        return (ax * self.gravX + ay * self.gravY + az * self.gravZ)

    def estimatedGravityDirection(self):
        self.gravX = 2 * (self.qx * self.qz - self.qw * self.qy)
        self.gravY = 2 * (self.qw * self.qx + self.qy * self.qz)
        self.gravZ = self.qw * self.qw - self.qx * self.qx - self.qy * self.qy + self.qz * self.qz

    def sensfusion6GetEulerRPY(self):
        gx = self.gravX
        gy = self.gravY
        gz = self.gravZ

        if (gx>1): 
            gx=1

        if (gx<-1):
            gx=-1

        self.yaw = np.atan2(2*(self.qw*self.qz + self.qx*self.qy), self.qw*self.qw + self.qx*self.qx - self.qy*self.qy - self.qz*self.qz) * 180 / np.pi
        self.pitch = np.arcsin(gx) * 180 / np.pi # Pitch seems to be inverted
        self.roll = np.atan2(gy, gz) * 180 / np.pi

    def sensfusion6GetAccZWithoutGravity(self, ax, ay, az):
        
        return self.sensfusion6GetAccZ(ax, ay, az) - self.baseZacc
        

def inv_sqrt(x):
    # Bit-level hacking via struct
    threehalfs = 1.5

    y = x
    i = struct.unpack('i', struct.pack('f', y))[0]   # float → int (32-bit)
    i = 0x5f3759df - (i >> 1)                        # magic number
    y = struct.unpack('f', struct.pack('i', i))[0]   # int → float (32-bit)

    # One iteration of Newton–Raphson
    y = y * (threehalfs - 0.5 * x * y * y)

    return y