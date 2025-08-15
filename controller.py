class PID_controller():
    def __init__(self, Kp, Ki, Kd, setpoint):
        
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0

    def compute(self, current_value, dt):
        
        error = self.setpoint - current_value

        P_out = self.Kp * error

        # Consider a different integral method
        self.integral += error * dt
        I_out = self.Ki * self.integral

        # First order differences for now
        derivative = (error - self.previous_error) / dt
        D_out = self.Kd * derivative

        # Compute output
        out = P_out + I_out + D_out

        # Error update
        self.previous_error = error

        return out

    