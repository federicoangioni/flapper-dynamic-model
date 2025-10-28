2025-09-02 22:58

Status: #mature

Tags: [[flapper]]


# flapper dynamic model

### Equations of Motion "template"

Due to the nice symmetry of the flapper all the product moments of inertia are zero kg/m².  Thus, the mass moment of inertia matrix $\mathcal{I}$ is diagonal. In fact, $I_{xy} = I_{yz} = 0$ as the flapper is perfectly symmetric about this planes, and $I_{xz} \sim 0$, as along the $xz$ plane the battery is slightly asymmetric, but the difference is really close to the axes and thus negligible.
$$
\mathcal{I} =
\begin{bmatrix}
I_{xx} & I_{xy} & I_{xz} \\
I_{yx} & I_{yy} & I_{yz} \\
I_{zx} & I_{zy} & I_{zz}
\end{bmatrix} = \begin{bmatrix}
I_{xx} & 0 & 0 \\
0 & I_{yy} & 0 \\
0 & 0 & I_{zz}
\end{bmatrix}
$$
Thus the torque equations of motion collapse into a reduced size with respect to what seen in [[translational and rotational damping of flapping flight]].

$$
\begin{align}
\dot{u} &= -(wq - vr) + \frac{X}{m} - g \sin \theta \\
\dot{v} &= -(ur - wp) + \frac{Y}{m} + g \cos \theta \sin \phi \\
\dot{w} &= -(vp - uq) + \frac{Z}{m} + g \cos \theta \cos \phi \\
I_{xx}\dot{p} &= (I_{yy} - I_{zz})qr + L \\
I_{yy}\dot{q} &= (I_{zz} - I_{xx})pr + M \\
I_{zz}\dot{r} &= (I_{xx} - I_{yy})pq + N \\
\dot{\phi} &= p + q \sin \phi \tan \theta + r \cos \phi \tan \theta \\
\dot{\theta} &= q \cos \phi - r \phi \\
\dot{\psi} &= r
\end{align}
$$
Where $X, Y, Z$ are the applied forces, and $L, M, N$ the torques applied.

From Matej's model, purely longitudinal.
$$
\begin{aligned}
X &= - (f_L + f_R)\:k_{xu}\:(u - l_z \;q + \dot{l_d}) \\
Y &= \varnothing \\
Z &= - (f_L + f_R) \:k_{zw}\:(w - l_d \: q) \\
L &= \varnothing\\
M &= X \cdot l_z + Z \cdot l_d\\
N &= \varnothing\\
\end{aligned}
$$

By applying the same philosophy of the damping being proportional to the flapping frequency and the velocity of the center of pressure we obtain:

Let's define some variables for simplicity:
$$
\begin{aligned}
Z_L & =f_L (w - l_w \sin(\Gamma)q + l_w \cos(\Gamma)p) \\
Z_R &= f_R (w - l_w \sin(\Gamma)q - l_w \cos(\Gamma)p) \\
l_d &= l_w \sin(\Gamma) \\
\sin(\alpha_L) &= \frac{-l_w\sin\Gamma + l_y\sin\Lambda}{l_k} \\
\sin(\alpha_R) &= \frac{-l_w\sin\Gamma - l_y\sin\Lambda}{l_k} \\
\end{aligned}
$$
Which becomes: (use $\Lambda$ for yaw angle actuator)

$$
\begin{aligned}
X &= -k_{xu} (f_L + f_R) (u - l_z q + l_w\dot{\Gamma}\sin{\Gamma}) - T_L\sin(\alpha_L) - T_R\sin(\alpha_R)\\
Y &= -k_{yv}(f_L + f_R)(v + l_z \:p)\\
Z &= -k_{zw} \left[Z_L + Z_R\right] - (T_R\cos(\alpha_R) + T_L\cos(\alpha_L))\\
\end{aligned}
$$





$$
\begin{aligned}
L &= -k_{zw} \left[Z_L - Z_R\right] l_w \cos(\Gamma)+ Y\:l_z + l_W\sin(\Gamma)(T_L\cos(\alpha_L) - T_R\cos(\alpha_R))\\
M &= -k_{xu} (f_L + f_R) (u - l_z q + l_w\dot{\Gamma}\sin{\Gamma}) \cdot l_z +(T_L\sin(\alpha_L) + T_R\sin(\alpha_R)) \cdot l_z-k_{zw} \left[Z_L + Z_R\right] \cdot l_d + (T_R\cos(\alpha_R) + T_L\cos(\alpha_L)) \cdot l_w\sin(\Gamma)\\
N &= -k_N \left[(f_L + f_R)Rr + (f_L - f_R)u + (f_L + f_R)\Gamma v\right] + l_w\cos(\Gamma)(T_R\sin(\alpha_R) - T_L\cos(\alpha_L))
\end{aligned}
$$
Understand what happens if dihedral and yaw servo are both engaged.

## TODO:
- [ ] Take a video from a lateral view, actuating first only the servo, then only the dihedral and then actuate first the dihedral and then the servo. 
- [ ] Take a video of the yaw servo actuator to find its transfer function, do the same for the dihedral angle, ideally also log the internal data
- [ ] Take all the necessary measures, on cad even better probably.

### Notes

$\Lambda > 0$ expect $r<0$ 




## Remarks
I have looked at Ernesto's paper once again, it seems like here the lift is not fully modeled here, however Matej did have some good results with this even up to 60$\degree$ pitch this is why for now I continued with this. I do think the damping coefficients should not be constant, as if we take a stroke averaged view of the flapper, we are effectively having a flat plate at different angles of attack, which means the damping will decrease with higher pitches/aoa (pitch=aoa almost), and effectively increase this "unmodeled", again to a certain extent, lift.

For sure from Ernesto's paper I don't think modelling the lift as $L=qSC_L$, is correct as there are different papers on wind tunnel experiments (see harvard's one) and analytical derivations which clearly show the direct proportionality with frequency and CoP speed.

Does Matej's "philosophy" of not modeling lift maybe work as while the angle of attack is decreasing the drag is decreasing and the lift increasing? 

I assume if the drone is hovering, and the yaw servo is actuated it actually slowly descends as part of the thrust vector is not dedicated anymore to the Z axis
## Questions

- can I have a look at the drone for a morning/afternoon (probably less) to take some measurements?
- How can I check what is the angle for  the yaw servo?
- See issues on GitHub.





# References

