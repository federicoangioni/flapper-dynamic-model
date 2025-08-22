from utils.filter import LP2Pfilter


def constrain(x, min_val, max_val):
    return max(min_val, min(x, max_val))


class PID_controller:
    def __init__(self, kp, ki, kd, kff, iLimit, dt, sampling_rate, cutoff_freq, enableDfilter, outputLimit=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.kff = kff
        self.sampling_rate = sampling_rate
        self.cutoff_freq = cutoff_freq
        self.deriv = 0
        self.dt = dt
        self.previous_measured = 0
        self.integ = 0
        self.iLimit = iLimit
        self.outputLimit = outputLimit
        self.enableDfilter = enableDfilter
        if enableDfilter:
            self.filt = LP2Pfilter(sampling_rate, cutoff_freq)

    def compute(self, measured, setpoint, is_yaw_angle: bool):
        error = setpoint - measured

        # Proportional output
        out_p = self.kp * error

        # Derivative of the measured process variable
        delta = -(measured - self.previous_measured)

        if is_yaw_angle:
            if delta > 180.0:
                delta -= 360.0
            elif delta < -180.0:
                delta += 360.0

        if not self.enableDfilter:
            deriv = delta / self.dt
        else:
            deriv = self.filt.apply(delta / self.dt)

        out_d = self.kd * deriv

        self.integ += error * self.dt

        if self.iLimit != 0:
            self.integ = constrain(self.integ, -self.iLimit, self.iLimit)

        out_i = self.ki * self.integ

        outFF = self.kff * setpoint

        # Compute output
        out = out_p + out_i + out_d + outFF

        if self.outputLimit != 0:
            out = constrain(out, -self.outputLimit, self.outputLimit)

        # Error update
        self.previous_measured = measured

        return out
