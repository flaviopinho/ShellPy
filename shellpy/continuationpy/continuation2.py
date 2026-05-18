import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla


class Continuation2:
    def __init__(self, model):
        self.model = model
        self.parameters = {
            'cont_max': 1000,
            'h_max': 2e-2,
            'h0': 1e-3,
            'h_min': 1e-6,
            'i_max': 6,
            'i_control': 3,
            'tol1': 1e-9,
            'tol2': 1e-9,
            'tol_rank': 1e-5,
            'index1': 0,
            'index2': -1,
            'jacobian_corrector': True,
            'min_param_component': 1e-3,
            'norm_load': 1.0,
            'determine_bifurcation': False,
            'determine_branch_point': False,
            'plot_real_time': False,
        }

        self.branches = {}
        self.ref_load_norm = 1.0

        if self.parameters.get('plot_real_time', False):
            self.fig = plt.figure(1)
            self.fig.add_subplot(111)
            plt.ion()

    def continue_branch(self, u0, t0, w0, branch_name=None):
        if branch_name is None:
            branch_name = "default"

        h = self.parameters['h0']
        w = w0
        u = u0.copy()
        t_u = t0
        jacobian_u = self.model['jacobian'](u, self.model)
        stability_u, _ = self.model['stability_check'](u, jacobian_u, self.model)

        u_load = np.zeros(np.shape(u0))
        u_load[self.parameters["index2"]] = self.parameters.get("norm_load", 1.0)
        load = self.model['residue'](u_load, self.model)

        self.ref_load_norm = np.linalg.norm(load, np.inf)
        if self.ref_load_norm < 1e-12:
            self.ref_load_norm = 1.0

        print(f"Norma de Carga de Referência: {self.ref_load_norm}")

        cont = 0
        while cont < self.parameters['cont_max']:
            if h < self.parameters['h_min']:
                print("-------- Tamanho do passo menor que o minimo. Encerrando. --------")
                break

            # 1. PASSO PREDITOR
            v = u + h * w * t_u

            converged = False

            # 2. PASSO CORRETOR (Newton no Sistema Expandido)
            for i in range(1, self.parameters['i_max'] + 1):
                residue = self.model['residue'](v, self.model)

                if self.parameters['jacobian_corrector'] or i == 1:
                    jacobian_v = self.model['jacobian'](v, self.model)
                else:
                    jacobian_v = jacobian_u

                # Equação de restrição do Pseudo-Arclength: (v - u)^T * t_u = h * w
                g = np.dot(v - u, t_u) - (h * w)

                # Montagem da Jacobiana Expandida (Matriz Quadrada n+1 x n+1)
                J_aug = np.vstack((jacobian_v, t_u))

                # Montagem do Lado Direito (Resíduos)
                rhs = np.append(-residue, -g)

                # ========================================================
                # INÍCIO DO RECONDICIONAMENTO (ROW SCALING)
                # ========================================================
                # 1. Encontra o valor máximo absoluto de cada linha
                escalas = np.max(np.abs(J_aug), axis=1)

                # 2. Previne divisão por zero (caso alguma linha seja nula)
                escalas[escalas == 0] = 1.0

                # 3. Divide cada linha da matriz e do vetor pelo seu respectivo fator
                # O 'np.newaxis' permite que o NumPy divida as colunas corretamente (broadcasting)
                J_cond = J_aug / escalas[:, np.newaxis]
                rhs_cond = rhs / escalas
                # ========================================================

                # Resolução do sistema com a matriz condicionada
                try:
                    dv = scipy.linalg.solve(J_cond, rhs_cond)
                except scipy.linalg.LinAlgError:
                    # Fallback
                    dv = np.linalg.lstsq(J_cond, rhs_cond, rcond=None)[0]

                v = v + dv

                error1 = np.linalg.norm(residue, np.inf) / self.ref_load_norm
                error2 = np.linalg.norm(dv, np.inf) / (np.linalg.norm(v, np.inf) + 1e-12)

                if error1 < self.parameters['tol1'] and error2 < self.parameters['tol2']:
                    converged = True
                    break

            # 3. TRATAMENTO DE FALHA OU SUCESSO DO CORRETOR
            if not converged:
                h /= 2
                print(f"-------- Corretor nao convergiu (passo {cont}). Reducao de passo para {h:e}. --------")
                continue  # Reinicia o while sem incrementar 'cont' e recalculando 'v' com o novo 'h'

            # A partir daqui, o passo foi um sucesso
            cont += 1
            stability_v, _ = self.model['stability_check'](v, jacobian_v, self.model)

            if stability_u * stability_v < 0:
                if self.parameters['determine_bifurcation']:
                    print("-------- Bifurcacao detectada. Calculando ponto exato... --------")
                    self.determining_bifurcation_point(u, v, branch_name)
                else:
                    print("-------- Bifurcacao detectada (calculo do ponto ignorado). --------")

            t_v = self.tangent_vector(jacobian_v)

            if np.dot(t_u, t_v) < 0:
                if self.parameters['determine_branch_point']:
                    print("-------- Ponto de ramificacao detectado. Calculando ponto exato... --------")
                    self.determining_branch_point(u, v, branch_name)
                else:
                    print("-------- Ponto de ramificacao detectado (calculo do ponto ignorado). --------")
                w = -w  # Inverte a direção se a tangente virou

            u = v
            t_u = t_v
            jacobian_u = jacobian_v
            stability_u = stability_v

            print(
                f"n={cont}   i={i}  erro1={error1:e}   error2={error2:e}   h={h:e}  estabilidade={stability_v:e}  Tipo: PR")

            self.save_results(branch_name, u, t_u, stability_u, "PR", cont)
            h = self.step_size_adjustment(i, h)

            if self.check_solution_boundary(u):
                print("-------- Limite de fronteira atingido. Encerrando continuação. --------")
                break

    def tangent_vector(self, jacobian, determine_f=False):
        n, m = jacobian.shape

        # A matriz tem dimensão n x (n+1). Computar a tangente via QR:
        Q, R = scipy.linalg.qr(jacobian.T, mode='full')

        # SUBSTITUÍDO: Evita o 'overflow' pegando apenas o sinal do determinante
        sign_Q, _ = np.linalg.slogdet(Q)
        sign_R, _ = np.linalg.slogdet(R[:n, :])

        sign_f = sign_Q * sign_R
        t = sign_f * Q[:, -1]

        if m == n + 1:
            t = self._ensure_parameter_progress(jacobian, t)

        if determine_f:
            # Retornamos apenas o sinal para a busca do ponto de bifurcação
            # (Isso transforma a interpolação secante em uma bisseção estável)
            return t, sign_f
        else:
            return t

    # (Os métodos determining_bifurcation_point, determining_branch_point,
    # branch_switching, save_results e etc permanecem inalterados ou podem
    # ser refatorados da mesma forma no futuro se houver gargalos neles).

    def save_results(self, branch_name, u, t_u, stability_u, point_type, cont):
        if branch_name not in self.branches:
            self.branches[branch_name] = []

        self.branches[branch_name].append({
            'u': u.copy(),
            't_u': t_u.copy(),
            'stability_check': stability_u,
            'point_type': point_type
        })

        if self.parameters.get('plot_real_time', False):
            if not hasattr(self, 'fig'):
                self.fig = plt.figure(1)
                self.fig.add_subplot(111)
                plt.ion()

            if self.model.get("output_function") is not None:
                u1, u2, label1, label2 = self.model["output_function"](u)
            else:
                u1 = u[self.parameters['index1']]
                u2 = u[self.parameters['index2']]
                label1 = f"u{self.parameters['index1']}"
                label2 = f"u{self.parameters['index2']}"

            marker = 'o' if point_type == "PR" else 's'
            color = 'red' if stability_u > 0 else ('blue' if point_type == "PR" else 'black')
            size = 2 if point_type == "PR" else 5
            label_text = '' if point_type == "PR" else point_type

            plt.figure(self.fig.number)
            ax = self.fig.axes[0]

            if len(self.branches[branch_name]) == 1:
                ax.set_title('Diagrama de bifurcação')
                ax.grid(True)
                ax.set_xlabel(label1)
                ax.set_ylabel(label2)

            if label_text:
                ax.text(u1 + 0.02, u2 + 0.02, label_text, color='black', fontsize=10)

            ax.plot(u1, u2, marker, color=color, markersize=size)
            if cont % 10 == 0:  # Atualiza o gráfico a cada 10 passos consolidados
                plt.draw()
                plt.pause(0.001)

    def step_size_adjustment(self, i, h):
        h = max(min(h * min([max([np.sqrt(self.parameters['i_control'] / i), 0.5]), 1.3]), self.parameters['h_max']),
                self.parameters['h_min'])
        return h

    def check_solution_boundary(self, u):
        if 'boundary' in self.model:
            u_min = self.model['boundary'][:, 0]
            u_max = self.model['boundary'][:, 1]
            if min(u_max - u) < 0 or min(u - u_min) < 0:
                return True
        return False

    def branch_switching_via_perturbation(self, u, J, num_tests=10):
        print("-------- Iniciando Branch Switching via Perturbacao --------")
        n = self.model.get('n', len(u) - 1)
        Jx = J[:n, :n]

        try:
            eigenvalues, eigenvectors = spla.eigs(Jx, k=1, which='SM')
            min_idx = np.argmin(np.abs(eigenvalues))
            critical_mode = np.real(eigenvectors[:, min_idx])

            critical_mode = critical_mode / np.linalg.norm(critical_mode, np.inf)

            perturbation = np.zeros_like(u)
            epsilon = self.parameters['h0'] * 10
            perturbation[:n] = epsilon * critical_mode

            return u + perturbation

        except Exception as e:
            print(f"Falha na perturbacao: {e}")
            return u

    def _ensure_parameter_progress(self, jacobian, tangent):
        min_param_component = self.parameters.get('min_param_component', 1e-3)
        if abs(tangent[-1]) >= min_param_component:
            return tangent

        J_u = jacobian[:, :-1]
        J_p = jacobian[:, -1]
        try:
            du = scipy.linalg.lstsq(J_u, -J_p, cond=None)[0]
            tangent_alt = np.hstack((du, np.array([1.0])))

            tangent_alt_norm = np.linalg.norm(tangent_alt, np.inf)
            if tangent_alt_norm > 0:
                tangent_alt /= tangent_alt_norm
                if np.dot(tangent_alt, tangent) < 0:
                    tangent_alt = -tangent_alt
                return tangent_alt
        except Exception:
            pass

        return tangent