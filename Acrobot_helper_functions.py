import numpy as np

def rk4(derivs, y0, t, lib):
    """
    Integrate 1-D or N-D system of ODEs batch-wise using 4-th order Runge-Kutta.

    Example for 2D system:

        >>> def derivs(x):
        ...     d1 =  x[0] + 2*x[1]
        ...     d2 =  -3*x[0] + 4*x[1]
        ...     return d1, d2

        >>> dt = 0.0005
        >>> t = np.arange(0.0, 2.0, dt)
        >>> y0 = (1,2)
        >>> yout = rk4(derivs, y0, t)

    Args:
        derivs: the derivative of the system and has the signature ``dy = derivs(yi)``
        y0: initial state vector
        t: sample times

    Returns:
        yout: Runge-Kutta approximation of the ODE
    """
    yout = lib.unsqueeze(y0, 1)  # batch_size x rk-steps x states

    for i in np.arange(len(t) - 1):
        this = t[i]
        dt = t[i + 1] - this
        dt2 = dt / 2.0
        y0 = yout[:, i, :]

        k1 = derivs(y0)
        k2 = derivs(y0 + dt2 * k1)
        k3 = derivs(y0 + dt2 * k2)
        k4 = derivs(y0 + dt * k3)
        y_next = lib.unsqueeze(y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), 1)
        yout = lib.concat([yout, y_next], 1)
    # We only care about the final timestep and we cleave off action value which will be zero
    return yout[:, -1, :4]



def _dsdt(parameters, lib, source='gym', book_or_nips='nips'):
    
    m1 = parameters.m1
    m2 = parameters.m2
    l1 = parameters.l1
    lc1 = parameters.lc1
    lc2 = parameters.lc2
    I1 = parameters.I1
    I2 = parameters.I2
    g = 9.8
    
    def _dsdt_gym(s_augmented):
        theta1, theta2, dtheta1, dtheta2, a = lib.unstack(s_augmented, 5, 1)
        d1 = (
            m1 * lc1**2
            + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * lib.cos(theta2))
            + I1
            + I2
        )
        d2 = m2 * (lc2**2 + l1 * lc2 * lib.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * lib.cos(theta1 + theta2 - lib.pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * lib.sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * lib.sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * lib.cos(theta1 - lib.pi / 2)
            + phi2
        )
        if book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2**2 + I2 - d2**2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (
                a
                + d2 / d1 * phi1
                - m2 * l1 * lc2 * dtheta1**2 * lib.sin(theta2)
                - phi2
            ) / (m2 * lc2**2 + I2 - d2**2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return lib.stack(
            [
                dtheta1,
                dtheta2,
                ddtheta1,
                ddtheta2,
                lib.zeros_like(dtheta1),
            ],
            1,
        )


    if source == 'mit':
        def M(theta1, theta2):
            c2 = lib.cos(theta2)

            a = I1 + I2 + m2 * l1 ** 2 + 2 * m2 * l1 * lc2 * c2
            b = I2 + m2 * l1 * lc2 * c2
            c = I2 + m2 * l1 * lc2 * c2
            d = I2

            M = lib.to_tensor(
                [
                    [a, b],
                    [c, d]
                ],
                lib.float32)

            return M

        def M_inv(thata1, theta2):
            c2 = lib.cos(theta2)

            a = I1 + I2 + m2 * l1 ** 2 + 2 * m2 * l1 * lc2 * c2
            b = I2 + m2 * l1 * lc2 * c2
            c = I2 + m2 * l1 * lc2 * c2
            d = lib.ones_like(c)*I2

            e = a*d-b*c

            if lib.less(lib.reduce_min(e, 0), 1.e-6):
                yellow_color_escape_code = '\033[93m'
                end_color_escape_code = '\033[0m'
                lib.print(f"{yellow_color_escape_code}Inertia matrix close to singular, minimal value {lib.reduce_min(e, 0)}, can result in nummerical instability{end_color_escape_code}")

            M_inv = lib.reciprocal(e) * lib.concat(
                [
                    lib.concat([d, -b], -1),
                    lib.concat([-c, a], -1),
                ],
                -2)

            return M_inv

        def C(theta1, theta2, dtheta1, dtheta2):
            s2 = lib.sin(theta2)

            a = -2 * m2 * l1 * lc2 * s2 * dtheta2
            b = -m2 * l1 * lc2 * s2 * dtheta2
            c = m2 * l1 * lc2 * s2 * dtheta1
            d = lib.zeros_like(c)

            C = lib.concat(
                [
                    lib.concat([a, b], -1),
                    lib.concat([c, d], -1)
                ],
                -2)

            return C

        def tau_g(theta1, theta2):
            s1 = lib.sin(theta1)
            s12 = lib.sin(theta1 + theta2)

            tau_g = lib.concat(
                [
                    -m1 * g * lc1 * s1 - m2 * g*(l1 * s1 + lc2 * s12),
                    -m2 * g * lc2 * s12
                ],
                -2)

            return tau_g

        B = lib.to_tensor([[0.0], [1.0]], lib.float32)[lib.newaxis, :, :]




    def _dsdt_mit(s_augmented):

        theta1, theta2, dtheta1, dtheta2, a = lib.unstack(lib.cast(s_augmented[:, lib.newaxis, lib.newaxis, :], lib.float32), 5, -1)

        q = lib.to_tensor([theta1, theta2], lib.float32)
        q_dot = lib.concat((dtheta1, dtheta2), -2)

        Control_matrix = lib.repeat(B, len(q_dot), 0)*a

        tau = tau_g(theta1, theta2)
        Cm = C(theta1, theta2, dtheta1, dtheta2)
        Cor = lib.matmul(Cm, q_dot)
        b = tau + Control_matrix - Cor
        q_dotdot = lib.matmul(M_inv(theta1, theta2), b)

        ddtheta1, ddtheta2 = lib.unstack(q_dotdot[:, lib.newaxis, :, :], 2, -2)

        s =  lib.concat(
            [
                dtheta1,
                dtheta2,
                ddtheta1,
                ddtheta2,
                lib.zeros_like(dtheta1),
            ],
            -1,
        )
        s = s[:, 0, :]

        return s

        
    
    if source == 'gym':
        return _dsdt_gym
    elif source == 'mit':
        return  _dsdt_mit
    else:
        raise KeyError('Source {} not recognised'.format(source))
        
