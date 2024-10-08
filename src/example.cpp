#include "lbfgs.hpp"
#include <iostream>
#include <iomanip>
#include <functional>
#include <chrono>

class Function {
public:
    /* f(x, y) = (1.0-x)^2+100.0*(y-x^2)^2 */
    double costFunction(const Eigen::VectorXd &x, Eigen::VectorXd &g) {
        const int n = x.size();
        double fx = 0.0;
        for (int i = 0; i < n; i += 2) {
            const double t1 = 1.0 - x(i);
            const double t2 = 10.0 * (x(i + 1) - x(i) * x(i));
            g(i + 1) = 20.0 * t2;
            g(i) = -2.0 * (x(i) * g(i + 1) + t1);
            fx += t1 * t1 + t2 * t2;
        }

        return fx;
    }
};

class Test {
public:

    void init(const int N) {
        double final_cost = 0.0;
        Eigen::VectorXd x(N);

        /* set the initial guess */
        x = 10.0 * Eigen::VectorXd::Random(N);

        /* Set teh function */
        std::shared_ptr<Function> func_ptr = std::make_shared<Function>();

        /* Set the minimization parameters */
        std::shared_ptr<LBFGS> lbfgs_ptr = std::make_shared<LBFGS>();

        /* Set the LBFGS parameters */
        lbfgs_ptr->param_ptr->grad_epsilon = 1.0e-9;
        lbfgs_ptr->param_ptr->delta = 1.0e-9;

        /* Set the callback data of lbfgs */
        lbfgs_ptr->cb_ptr->proc_evaluate =
            std::bind(&Function::costFunction, func_ptr, std::placeholders::_1, std::placeholders::_2);

        /* Start optimization */
        int val = lbfgs_ptr->optimize(x, final_cost);

                /* Report the result. */
        std::cout << std::setprecision(3)
                  << "================================" << std::endl
                  << "L-BFGS Optimization Returned: " << val << std::endl
                  << "Minimized Cost: " << final_cost << std::endl
                  << "Optimal Variables: " << std::endl
                  << x.transpose() << std::endl;

        std::cout << lbfgs_ptr->display_message(val) << std::endl;
    }
};

int main(int argc, char **argv)
{
    auto start = std::chrono::steady_clock::now();
    std::shared_ptr<Test> test_ptr = std::make_shared<Test>();
    test_ptr->init(10000);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Time: " << elapsed_seconds.count() << "s" << std::endl;

    return 0;
}
