#ifndef LBFGS_HPP
#define LBFGS_HPP

#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include <functional>

#include <Eigen/Dense>

class LBFGS {
public:
    // L-BFGS optimization parameters
    class Parameter {
    public:
        int mem_size = 8; // The number of corrections to approximate the inverse hessian matrix.
        double grad_epsilon = 1.0e-6; // Epsilon for gradient convergence test.
        int past = 3; // Distance for delta-based convergence test.
        double delta = 1.0e-6; // Delta for convergence test.
        int max_iterations = 0; // The maximum number of iterations.
        int max_linesearch = 64; // The maximum number of trials for the line search.
        double min_step = 1.0e-20; // The minimum step of the line search routine.
        double max_step = 1.0e+20; // The maximum step of the line search routine.
        double f_dec_coeff = 1.0e-6; // The coefficient for the accuracy of the line search routine.
        double s_curv_coeff = 0.9; // A parameter to control the accuracy of the line search routine.
        double cautious_factor = 1.0e-6; // A parameter to ensure the global convergence for nonconvex functions.
        double machine_prec = 1.0e-16; // The machine precision for floating-point values.
    };
    std::shared_ptr<Parameter> param_ptr;

    // L-BFGS optimization return codes. If occurs errors will return a negtive value.
    enum Code {
        /** L-BFGS reaches convergence. */
        LBFGS_CONVERGENCE = 0,
        /** L-BFGS satisfies stopping criteria. */
        LBFGS_STOP,

        /** Unknown error. */
        LBFGSERR_UNKNOWNERROR = -1024,
        /** Invalid number of variables specified. */
        LBFGSERR_INVALID_N,
        /** Invalid parameter lbfgs_parameter_t::mem_size specified. */
        LBFGSERR_INVALID_MEMSIZE,
        /** Invalid parameter lbfgs_parameter_t::g_epsilon specified. */
        LBFGSERR_INVALID_GEPSILON,
        /** Invalid parameter lbfgs_parameter_t::past specified. */
        LBFGSERR_INVALID_TESTPERIOD,
        /** Invalid parameter lbfgs_parameter_t::delta specified. */
        LBFGSERR_INVALID_DELTA,
        /** Invalid parameter lbfgs_parameter_t::min_step specified. */
        LBFGSERR_INVALID_MINSTEP,
        /** Invalid parameter lbfgs_parameter_t::max_step specified. */
        LBFGSERR_INVALID_MAXSTEP,
        /** Invalid parameter lbfgs_parameter_t::f_dec_coeff specified. */
        LBFGSERR_INVALID_FDECCOEFF,
        /** Invalid parameter lbfgs_parameter_t::s_curv_coeff specified. */
        LBFGSERR_INVALID_SCURVCOEFF,
        /** Invalid parameter lbfgs_parameter_t::machine_prec specified. */
        LBFGSERR_INVALID_MACHINEPREC,
        /** Invalid parameter lbfgs_parameter_t::max_linesearch specified. */
        LBFGSERR_INVALID_MAXLINESEARCH,
        /** The function value became NaN or Inf. */
        LBFGSERR_INVALID_FUNCVAL,
        /** The line-search step became smaller than lbfgs_parameter_t::min_step. */
        LBFGSERR_MINIMUMSTEP,
        /** The line-search step became larger than lbfgs_parameter_t::max_step. */
        LBFGSERR_MAXIMUMSTEP,
        /** Line search reaches the maximum, assumptions not satisfied or precision not achievable.*/
        LBFGSERR_MAXIMUMLINESEARCH,
        /** The algorithm routine reaches the maximum number of iterations. */
        LBFGSERR_MAXIMUMITERATION,
        /** Relative search interval width is at least lbfgs_parameter_t::machine_prec. */
        LBFGSERR_WIDTHTOOSMALL,
        /** A logic error (negative line-search step) occurred. */
        LBFGSERR_INVALIDPARAMETERS,
        /** The current search direction increases the cost function value. */
        LBFGSERR_INCREASEGRADIENT,
    };

    /**
    Callback interface to provide cost function and gradient evaluations.

    @param x: The current values of variables.
    @param g: The gradient vector. The callback function must compute the gradient values for the current variables.

    @ return double: The value of the cost function for the current variables.
    */
    using evaluate = std::function<double(const Eigen::VectorXd &x, Eigen::VectorXd &g)>;

    /**
        Callback interface to provide an upper bound at the beginning of the current line search.

        @param xp: The values of variables before current line search.
        @param d: The step vector. It can be the descent direction.

        @return double: The upperboud of the step in current line search routine,
                        such that (stpbound * d) is the maximum reasonable step.
    */
    using stepbound = std::function<double(const Eigen::VectorXd &xp, const Eigen::VectorXd &d)>;

    // Callback class
    class Callback {
    public:
        evaluate proc_evaluate;
        stepbound proc_stepbound;
    };
    std::shared_ptr<Callback> cb_ptr;

    /**
        Line search method for smooth or nonsmooth functions.

        @see
            Adrian S. Lewis and Michael L. Overton. Nonsmooth optimization via quasi-Newton methods.
            Mathematical Programming, Vol 141, No 1, pp. 135-163, 2013.
    */
    int line_search_lewisoverton(Eigen::VectorXd &x,
                                 double &f,
                                 Eigen::VectorXd &g,
                                 double &stp,
                                 const Eigen::VectorXd &s,
                                 const Eigen::VectorXd &xp,
                                 const Eigen::VectorXd &gp,
                                 const double stpmin,
                                 const double stpmax) {
        int count = 0;
        bool brackt = false, touched = false;
        double finit, dginit, dgtest, dstest;
        double mu = 0.0, nu = stpmax;

        /* Check the input parameters for errors. */
        if (!(stp > 0.0)) {
            return LBFGSERR_INVALIDPARAMETERS;
        }

        /* Compute the initial gradient in the search direction. */
        dginit = gp.dot(s);

        /* Make sure that s points to a descent direction. */
        if (0.0 < dginit) {
            return LBFGSERR_INCREASEGRADIENT;
        }

        /* The initial value of the cost function. */
        finit = f;
        dgtest = param_ptr->f_dec_coeff * dginit;
        dstest = param_ptr->s_curv_coeff * dginit;

        while (true) {
            x = xp + stp * s;

            /* Evaluate the function and gradient values. */
            f = cb_ptr->proc_evaluate(x, g);
            ++count;

            /* Test for errors. */
            if (std::isinf(f) || std::isnan(f)) {
                return LBFGSERR_INVALID_FUNCVAL;
            }

            /* Check the Armijo condition. */
            if (f > finit + stp * dgtest) {
                nu = stp;
                brackt = true;
            } else {
                /* Check the weak Wolfe condition. */
                if (g.dot(s) < dstest) {
                    mu = stp;
                } else {
                    return count;
                }
            }
            if (param_ptr->max_linesearch <= count) {
                /* Maximum number of iteration. */
                return LBFGSERR_MAXIMUMLINESEARCH;
            }
            if (brackt && (nu - mu) < param_ptr->machine_prec * nu) {
                /* Relative interval width is at least machine_prec. */
                return LBFGSERR_WIDTHTOOSMALL;
            }

            if (brackt) {
                stp = 0.5 * (mu + nu);
            } else {
                stp *= 2.0;
            }

            if (stp < stpmin) {
                /* The step is the minimum value. */
                return LBFGSERR_MINIMUMSTEP;
            }
            if (stp > stpmax) {
                if (touched) {
                    /* The step is the maximum value. */
                    return LBFGSERR_MAXIMUMSTEP;
                } else {
                    /* The maximum value should be tried once. */
                    touched = true;
                    stp = stpmax;
                }
            }
        }
    }

    /**
        Start a L-BFGS optimization.

        @param x: The vector of decision variables.
                  THE INITIAL GUESS x0 SHOULD BE SET BEFORE THE CALL!
                  A client program can receive decision variables
                  through this vector, at which the cost and its
                  gradient are queried during minimization.
        @param f: The ref to the variable that receives the final
                  value of the cost function for the variables.
        @param instance: A user data pointer for client programs. The callback functions will receive the value of this
                         argument.
        @param param: The parameters for L-BFGS optimization.
        @return int: The status code. This function returns a nonnegative integer if the minimization process terminates
                    without an error. A negative integer indicates an error.
    */
    int optimize(Eigen::VectorXd &x, double &f) {
        int ret, i, j, k, ls, end, bound;
        double step, step_min, step_max, fx, ys, yy;
        double gnorm_inf, xnorm_inf, beta, rate, cau;

        const int n = x.size();
        const int m = param_ptr->mem_size;

        /* Check the input parameters for errors. */
        if (n <= 0) {
            return LBFGSERR_INVALID_N;
        }
        if (m <= 0) {
            return LBFGSERR_INVALID_MEMSIZE;
        }
        if (param_ptr->grad_epsilon < 0.0) {
            return LBFGSERR_INVALID_GEPSILON;
        }
        if (param_ptr->past < 0) {
            return LBFGSERR_INVALID_TESTPERIOD;
        }
        if (param_ptr->delta < 0.0) {
            return LBFGSERR_INVALID_DELTA;
        }
        if (param_ptr->min_step < 0.0) {
            return LBFGSERR_INVALID_MINSTEP;
        }
        if (param_ptr->max_step < param_ptr->min_step) {
            return LBFGSERR_INVALID_MAXSTEP;
        }
        if (!(param_ptr->f_dec_coeff > 0.0 && param_ptr->f_dec_coeff < 1.0)) {
            return LBFGSERR_INVALID_FDECCOEFF;
        }
        if (!(param_ptr->s_curv_coeff < 1.0 && param_ptr->s_curv_coeff > param_ptr->f_dec_coeff)) {
            return LBFGSERR_INVALID_SCURVCOEFF;
        }
        if (!(param_ptr->machine_prec > 0.0)) {
            return LBFGSERR_INVALID_MACHINEPREC;
        }
        if (param_ptr->max_linesearch <= 0) {
            return LBFGSERR_INVALID_MAXLINESEARCH;
        }

        /* Prepare intermediate variables. */
        Eigen::VectorXd xp(n);
        Eigen::VectorXd g(n);
        Eigen::VectorXd gp(n);
        Eigen::VectorXd d(n);
        Eigen::VectorXd pf(std::max(1, param_ptr->past));

        /* Initialize the limited memory. */
        Eigen::VectorXd lm_alpha = Eigen::VectorXd::Zero(m);
        Eigen::MatrixXd lm_s = Eigen::MatrixXd::Zero(n, m);
        Eigen::MatrixXd lm_y = Eigen::MatrixXd::Zero(n, m);
        Eigen::VectorXd lm_ys = Eigen::VectorXd::Zero(m);

        /* Evaluate the function value and its gradient. */
        fx = cb_ptr->proc_evaluate(x, g);

        /* Store the initial value of the cost function. */
        pf(0) = fx;

        /* Compute the directory. We assume the initial hessian matrix H_0 as the identity matrix. */
        d = -g;

        /* Make sure that the initial variables are not a stationary point. */
        gnorm_inf = g.cwiseAbs().maxCoeff();
        xnorm_inf = x.cwiseAbs().maxCoeff();

        if (gnorm_inf / std::max(1.0, xnorm_inf) <= param_ptr->grad_epsilon) {
            /* The initial guess is already a stationary point. */
            ret = LBFGS_CONVERGENCE;
        } else {
            /* Compute the initial step */
            step = 1.0 / d.norm();

            k = 1;
            end = 0;
            bound = 0;

            while (true) {
                /* Store the current position and gradient vectors. */
                xp = x;
                gp = g;

                /* If the step bound can be provided dynamically, then apply it. */
                step_min = param_ptr->min_step;
                step_max = param_ptr->max_step;
                if (cb_ptr->proc_stepbound) {
                    step_max = cb_ptr->proc_stepbound(xp, d);
                    step_max = std::min(step_max, param_ptr->max_step);
                }
                step = step < step_max ? step : 0.5 * step_max;

                /* Search for an optimal step. */
                ls = line_search_lewisoverton(x, fx, g, step, d, xp, gp, step_min, step_max);

                if (ls < 0) {
                    /* Revert to the previous point. */
                    x = xp;
                    g = gp;
                    ret = ls;
                    break;
                }

                /*
                Convergence test.
                The criterion is given by the following formula: ||g(x)||_inf / max(1, ||x||_inf) < g_epsilon
                */
                gnorm_inf = g.cwiseAbs().maxCoeff();
                xnorm_inf = x.cwiseAbs().maxCoeff();
                if (gnorm_inf / std::max(1.0, xnorm_inf) < param_ptr->grad_epsilon) {
                    /* Convergence. */
                    ret = LBFGS_CONVERGENCE;
                    break;
                }

                /*
                Test for stopping criterion.
                The criterion is given by the following formula: |f(past_x) - f(x)| / max(1, |f(x)|) < \delta.
                */
                if (0 < param_ptr->past) {
                    /* We don't test the stopping criterion while k < past. */
                    if (param_ptr->past <= k) {
                        /* The stopping criterion. */
                        rate = std::abs(pf(k % param_ptr->past) - fx) / std::max(1.0, std::abs(fx));

                        if (rate < param_ptr->delta) {
                            ret = LBFGS_STOP;
                            break;
                        }
                    }

                    /* Store the current value of the cost function. */
                    pf(k % param_ptr->past) = fx;
                }

                if (param_ptr->max_iterations != 0 && param_ptr->max_iterations <= k) {
                    /* Maximum number of iterations. */
                    ret = LBFGSERR_MAXIMUMITERATION;
                    break;
                }

                /* Count the iteration number. */
                ++k;

                /*
                Update vectors s and y:
                s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
                y_{k+1} = g_{k+1} - g_{k}.
                */
                lm_s.col(end) = x - xp;
                lm_y.col(end) = g - gp;

                /*
                Compute scalars ys and yy:
                ys = y^t \cdot s = 1 / \rho.
                yy = y^t \cdot y.
                Notice that yy is used for scaling the hessian matrix H_0 (Cholesky factor).
                */
                ys = lm_y.col(end).dot(lm_s.col(end));
                yy = lm_y.col(end).squaredNorm();
                lm_ys(end) = ys;

                /* Compute the negative of gradients. */
                d = -g;

                /*
                Only cautious update is performed here as long as
                (y^t \cdot s) / ||s_{k+1}||^2 > \epsilon * ||g_{k}||^\alpha,
                where \epsilon is the cautious factor and a proposed value
                for \alpha is 1.
                This is not for enforcing the PD of the approxomated Hessian
                since ys > 0 is already ensured by the weak Wolfe condition.
                This is to ensure the global convergence as described in:
                Dong-Hui Li and Masao Fukushima. On the global convergence of
                the BFGS method for nonconvex unconstrained optimization problems.
                SIAM Journal on Optimization, Vol 11, No 4, pp. 1054-1064, 2011.
                */
                cau = lm_s.col(end).squaredNorm() * gp.norm() * param_ptr->cautious_factor;

                if (ys > cau) {
                    /*
                    Recursive formula to compute dir = -(H \cdot g).
                    This is described in page 779 of:
                    Jorge Nocedal.
                    Updating Quasi-Newton Matrices with Limited Storage. Mathematics of Computation, Vol. 35, No. 151,
                    pp. 773--782, 1980.
                    */
                    ++bound;
                    bound = std::min(bound, m);
                    end = (end + 1) % m;

                    j = end;
                    for (i = 0; i < bound; ++i) {
                        j = (j + m - 1) % m; /* if (--j == -1) j = m-1; */
                        /* \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}. */
                        lm_alpha(j) = lm_s.col(j).dot(d) / lm_ys(j);
                        /* q_{i} = q_{i+1} - \alpha_{i} y_{i}. */
                        d += (-lm_alpha(j)) * lm_y.col(j);
                    }

                    d *= ys / yy;

                    for (i = 0; i < bound; ++i)
                    {
                        /* \beta_{j} = \rho_{j} y^t_{j} \cdot \gamm_{i}. */
                        beta = lm_y.col(j).dot(d) / lm_ys(j);
                        /* \gamm_{i+1} = \gamm_{i} + (\alpha_{j} - \beta_{j}) s_{j}. */
                        d += (lm_alpha(j) - beta) * lm_s.col(j);
                        j = (j + 1) % m; /* if (++j == m) j = 0; */
                    }
                }

                /* The search direction d is ready. We try step = 1 first. */
                step = 1.0;
            }
        }

        /* Return the final value of the cost function. */
        f = fx;

        return ret;
    }

    /**
        Get string description of an optimize() return code.

        @param err: A value returned by optimize().
        @return const std::string: The string description of the return code.
    */
    const std::string display_message(const int err) {
        switch (err) {
            case LBFGS_CONVERGENCE:
                return "Success: reached convergence (g_epsilon).";

            case LBFGS_STOP:
                return "Success: met stopping criteria (past f decrease less than delta).";

            case LBFGSERR_UNKNOWNERROR:
                return "Unknown error.";

            case LBFGSERR_INVALID_N:
                return "Invalid number of variables specified.";

            case LBFGSERR_INVALID_MEMSIZE:
                return "Invalid parameter lbfgs_parameter_t::mem_size specified.";

            case LBFGSERR_INVALID_GEPSILON:
                return "Invalid parameter lbfgs_parameter_t::g_epsilon specified.";

            case LBFGSERR_INVALID_TESTPERIOD:
                return "Invalid parameter lbfgs_parameter_t::past specified.";

            case LBFGSERR_INVALID_DELTA:
                return "Invalid parameter lbfgs_parameter_t::delta specified.";

            case LBFGSERR_INVALID_MINSTEP:
                return "Invalid parameter lbfgs_parameter_t::min_step specified.";

            case LBFGSERR_INVALID_MAXSTEP:
                return "Invalid parameter lbfgs_parameter_t::max_step specified.";

            case LBFGSERR_INVALID_FDECCOEFF:
                return "Invalid parameter lbfgs_parameter_t::f_dec_coeff specified.";

            case LBFGSERR_INVALID_SCURVCOEFF:
                return "Invalid parameter lbfgs_parameter_t::s_curv_coeff specified.";

            case LBFGSERR_INVALID_MACHINEPREC:
                return "Invalid parameter lbfgs_parameter_t::machine_prec specified.";

            case LBFGSERR_INVALID_MAXLINESEARCH:
                return "Invalid parameter lbfgs_parameter_t::max_linesearch specified.";

            case LBFGSERR_INVALID_FUNCVAL:
                return "The function value became NaN or Inf.";

            case LBFGSERR_MINIMUMSTEP:
                return "The line-search step became smaller than lbfgs_parameter_t::min_step.";

            case LBFGSERR_MAXIMUMSTEP:
                return "The line-search step became larger than lbfgs_parameter_t::max_step.";

            case LBFGSERR_MAXIMUMLINESEARCH:
                return "Line search reaches the maximum try number, assumptions not satisfied or precision not "
                       "achievable.";

            case LBFGSERR_MAXIMUMITERATION:
                return "The algorithm routine reaches the maximum number of iterations.";

            case LBFGSERR_WIDTHTOOSMALL:
                return "Relative search interval width is at least lbfgs_parameter_t::machine_prec.";

            case LBFGSERR_INVALIDPARAMETERS:
                return "A logic error (negative line-search step) occurred.";

            case LBFGSERR_INCREASEGRADIENT:
                return "The current search direction increases the cost function value.";

            default:
                return "(unknown)";
        }
    }

    LBFGS() {
        param_ptr = std::make_shared<Parameter>();
        cb_ptr = std::make_shared<Callback>();
    }
    ~LBFGS() {
        param_ptr.reset();
        cb_ptr.reset();
    };

};

#endif // LBFGS_HPP
