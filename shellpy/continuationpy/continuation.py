import numpy as np
import matplotlib.pyplot as plt
import scipy


class Continuation:
    def __init__(self, model):
        self.model = model
        self.parameters = {

            'cont_max': 1000,  # Número máximo de iterações
            'h_max': 2e-2,  # Tamanho mínimo do passo
            'h0': 1e-3,  # Tamanho inicial do passo
            'h_min': 1e-6,  # Tamanho mínimo do passo
            'i_max': 6,  # Número máximo de iterações no corretor
            'i_control': 3,
            'tol1': 1e-9,  # Tolerância 1 para o erro do residuo
            'tol2': 1e-9,  # Tolerância 2 para o erro da correção
            'tol_rank': 1e-5,
            'index1': 0,
            'index2': -1
        }

        self.branches = {}  # Dicionário para armazenar os ramos com seus pontos

        # Inicializa o gráfico
        self.fig = plt.figure(1)
        self.fig.add_subplot(111)


    def continue_branch(self, u0, t0, w0, branch_name=None):

        if branch_name is None:
            branch_name = "default"

        h = self.parameters['h0']
        w = w0
        u = u0
        t_u = t0
        jacobian_u = self.model['jacobian'](u, self.model)
        stability_u, _ = self.model['stability_check'](u, jacobian_u, self.model)

        for cont in range(self.parameters['cont_max']):
            if h < self.parameters['h_min']:
                print("-------- Tamanho do passo menor que o minimo. --------")
                break

            # Preditor
            v = u + h * w * t_u

            for i in range(1, self.parameters['i_max'] + 1):
                # Corretor
                residue = self.model['residue'](v, self.model)
                jacobian_v = self.model['jacobian'](v, self.model)
                dv = np.linalg.pinv(jacobian_v) @ residue
                # dv = np.linalg.lstsq(jacobian_v, residue, rcond=None)[0]

                v = v - dv

                error1 = np.linalg.norm(residue, 2)
                error2 = np.linalg.norm(dv, 2) / np.linalg.norm(v, 2)

                if error1 < self.parameters['tol1'] or error2 < self.parameters['tol2']:
                    break

                if i == self.parameters['i_max']:
                    v = u
                    h /= 2
                    print("-------- Corretor nao convergiu. Reducao de passo. --------")
                    continue

            # Verificar bifurcações
            stability_v, _ = self.model['stability_check'](v, jacobian_v, self.model)

            if stability_u * stability_v < 0:
                print("-------- Bifurcacao. --------")
                self.determining_bifurcation_point(u, v, branch_name)

            # Verificar ponto de ramificação
            t_v = self.tangent_vector(jacobian_v)

            if np.dot(t_u, t_v) < 0:
                print("-------- Ponto de ramificacao. --------")
                self.determining_branch_point(u, v, branch_name)
                w = -w

            # Atualização das variáveis para o próximo ponto da curva
            u = v
            t_u = t_v
            jacobian_u = jacobian_v
            stability_u = stability_v

            self.model['u'] = u

            # Impressão de resultados parciais em tela
            print(
                f"n={cont}   i={i}  erro1={error1:e}   error2={error2:e}   h={h:e}  estabilidade={stability_v:e}  Tipo: PR")

            # Saída dos resultados
            self.save_results(branch_name, u, t_u, stability_u, "PR")

            # Ajustar tamanho do passo
            h = self.step_size_adjustment(i, h)

            # Verificar intervalos das variáveis para finalização do loop
            if self.check_solution_boundary(u):
                break

    def tangent_vector(self, jacobian, determine_f=False):
        n, m = jacobian.shape

        Q, R = np.linalg.qr(jacobian.T, mode='complete')
        det_Q = np.linalg.det(Q)
        det_R = np.linalg.det(R[:n, :])
        f = det_Q * det_R
        t = np.sign(f) * Q[:, -1]

        if determine_f:
            return t, f
        else:
            return t

    def determining_bifurcation_point(self, u1, u2, branch_name):
        jacobian_u1 = self.model['jacobian'](u1, self.model)
        jacobian_u2 = self.model['jacobian'](u2, self.model)

        stability_u1, type1 = self.model['stability_check'](u1, jacobian_u1, self.model)
        stability_u2, type2 = self.model['stability_check'](u2, jacobian_u2, self.model)

        if stability_u1 > 0:
            tipo = type1
            stability_um = stability_u1
            um = u1
        else:
            tipo = type2
            stability_um = stability_u2
            um = u2

        u10 = u1
        u20 = u2
        stability_10 = stability_u1
        stability_20 = stability_u2

        try:
            for j in range(10):
                fac = stability_u1 / (stability_u2 - stability_u1)
                if abs(fac) > 1:
                    fac = np.sign(fac) * 0.5

                um = u1 - fac * (u2 - u1)

                for i in range(1, self.parameters['i_max'] + 1):
                    # Corretor
                    residue = self.model['residue'](um)
                    jacobian_um = self.model['jacobian'](um)
                    dum = np.linalg.pinv(jacobian_um) @ residue
                    # dum = np.linalg.lstsq(jacobian_um, residue, rcond=None)[0]  # Resolução do sistema linear

                    um = um - dum

                    error1 = np.linalg.norm(residue, 2)
                    error2 = np.linalg.norm(dum, 2) / np.linalg.norm(um, 2)

                    if error1 < self.parameters['tol1'] or error2 < self.parameters['tol2']:
                        break

                # Estabilidade
                stability_um, _ = self.model['stability_check'](um, jacobian_um, self.model)

                if abs(stability_um) < self.parameters['tol1']:
                    break

                if stability_u1 * stability_um < 0:
                    u2 = um
                    stability_u2 = stability_um
                elif stability_u2 * stability_um < 0:
                    u1 = um
                    stability_u1 = stability_um
        except:
            fac = stability_10 / (stability_20 - stability_10)
            um = u10 - fac * (u20 - u10)
            jacobian_um = self.model['jacobian'](um)

        tm = self.tangent_vector(jacobian_um)
        self.save_results(branch_name, um, tm, stability_um, tipo)

    def determining_branch_point(self, u, v, branch_name):
        u1 = u
        u2 = v

        J1 = self.model['jacobian'](u1)
        J2 = self.model['jacobian'](u2)

        t1, f_1 = self.tangent_vector(J1, True)
        t2, f_2 = self.tangent_vector(J2, True)

        u10 = u1
        u20 = u2
        f_10 = f_1
        f_20 = f_2

        try:
            for j in range(10):
                fac = f_1 / (f_2 - f_1)
                if abs(fac) > 0.8:
                    fac = np.sign(fac) * 0.5

                um = u1 - fac * (u2 - u1)

                for i in range(1, self.parameters['i_max'] + 1):
                    # Corretor
                    H = self.model['residue'](um)
                    J_um = self.model['jacobian'](um)
                    dum = np.linalg.pinv(J_um) @ H
                    # dum = np.linalg.lstsq(J_um, H, rcond=None)[0]  # Solução de mínimos quadrados
                    um -= dum

                    erro1 = np.linalg.norm(H, 2)
                    erro2 = np.linalg.norm(dum, 2) / np.linalg.norm(um, 2)

                    if erro1 < self.parameters['tol1'] or erro2 < self.parameters['tol2']:
                        break

                _, f_m = self.tangent_vector(J_um, True)

                if abs(f_m) < self.parameters['tol1']:
                    break

                if f_1 * f_m < 0:
                    u2 = um
                    f_2 = f_m
                elif f_2 * f_m < 0:
                    u1 = um
                    f_1 = f_m

        except:
            fac = f_10 / (f_20 - f_10)
            um = u10 - fac * (u20 - u10)
            J_um = self.model['jacobian'](um)

        try:
            rank_J_um = np.linalg.matrix_rank(J_um, tol=self.parameters['tol_rank'])
            stability_um = self.model['stability_check'](um, J_um, self.model)[0]
            if J_um.shape[1] - rank_J_um == 2:
                tipo = 'BPS'  # Ponto de ramificação simples
            else:
                tipo = 'BPC'  # Ponto de ramificação complexo
        except:
            tipo = 'BPC'
            stability_um = 0



        self.save_results(branch_name, um, t1, stability_um, tipo)

    def save_results(self, branch_name, u, t_u, stability_u, point_type):
        # Armazenar resultados no branch
        if branch_name not in self.branches:
            self.branches[branch_name] = []

        self.branches[branch_name].append({
            'u': u,
            't_u': t_u,
            'stability_check': stability_u,
            'point_type': point_type
        })

        if self.model.get("output_function") is not None:
            u1, u2, label1, label2 = self.model["output_function"](u)
        else:
            u1 = u[self.parameters['index1']]
            u2 = u[self.parameters['index2']]
            label1 = f'u{self.parameters['index1']}'
            label2 = f'u{self.parameters['index2']}'

        # Gráfico da evolução de u


        # Se for um ponto de bifurcação ou de ramificação, fazer o ponto vermelho
        if point_type == "PR":
            marker = 'o'
            if stability_u > 0:
                color = 'red'
            else:
                color = 'blue'
            label = ''
            size = 2  # Diminuindo o tamanho do ponto
        else:
            marker = 's'
            color = 'black'
            label = point_type
            size = 5  # Diminuindo o tamanho do ponto

        plt.figure(self.fig.number)
        n = len(self.fig.axes)

        ax = plt.subplot(1, n, 1)
        ax.set_title('Diagrama de bifurcação')
        ax.grid(True)

        if label:
            ax.text(u1 + 0.02, u2 + 0.02, label, color='black', fontsize=10)

        ax.set_xlabel(label1)
        ax.set_ylabel(label2)

        # Plotar o ponto com a cor e tamanho definidos
        ax.plot(u1, u2, marker, color=color, markersize=size)

        plt.tight_layout()  # Adjust layout to prevent labels from overlapping

        # Atualizar o gráfico
        plt.pause(0.01)  # Pausa para permitir atualização dinâmica

    def step_size_adjustment(self, i, h):
        h = max(min(h * min([max([np.sqrt(self.parameters['i_control'] / i), 0.5]), 1.3]), self.parameters['h_max']),
                self.parameters['h_min'])
        return h

    def check_solution_boundary(self, u):
        u_min = self.model['boundary'][:, 0]
        u_max = self.model['boundary'][:, 1]

        if min(u_max - u) < 0 or min(u - u_min) < 0:
            return True
        else:
            return False

    def branch_switching(self, u, tol=None):

        J = self.model['jacobian'](u)

        if tol is None:
            tol = self.parameters['tol_rank']

        M = scipy.linalg.null_space(J, rcond=tol)  # Using scipy's null_space function

        if M.shape[1] == 2:
            t1, t2 = M[:, 0], M[:, 1]
            return t1, t2
        else:
            return self.branch_switching_via_perturbation(u)

    def branch_switching_via_perturbation(self, u, tol=None, num_tests=10):
        pass
