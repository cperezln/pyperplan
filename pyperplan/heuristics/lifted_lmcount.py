from networkx.algorithms.smallworld import omega
from scipy.stats import alpha


class LiftedLMCountHeuristic:

    def __init__(self, task, lgg, probabilities, restriction, types):
        self.iter = 0
        self.task = task
        self.lgg = lgg
        self.types = types
        self.costs = {'(' + ' '.join([k[0]] + [i[0] for i in k[1:]]) + ')': v for k, v in probabilities.items()}
        self.alpha_dict = dict()
        self.restriction = restriction
        self.landmarks = set()
        for k, v in lgg.items():
            self.landmarks.add(k)
            for ele in v:
                self.landmarks.add(ele)

    def _check_disj_pred(self, pred, number=False):
        if not number:
            for obj in pred.replace(')', '').replace('(', '').split(' ')[1:]:
                if '?' in obj:
                    return True
            return False
        else:
            var_counter = 0
            for obj in pred[1:]:
                if '?' in obj:
                    var_counter += 1
            return var_counter

    def _check_in(self, pred, set):
        if not self._check_disj_pred(pred):
            return pred in set
        else:
            for grounded in set:
                act_pred = pred.replace(')', '').split(' ')
                act_ground = grounded.replace(')', '').split(' ')
                check_names = act_pred[0] == act_ground[0]
                if check_names:
                    check = sum([
                        1 for p, q in zip(act_ground[1:], act_pred[1:])
                        if (p == q or ('?' in q and p not in self.restriction[q]))
                    ])
                    if check == len(act_ground[1:]):
                        if pred not in self.alpha_dict:
                            self.alpha_dict[pred] = self.tomega(act_ground, act_pred)
                        return True
        return False

    def tomega(self, node_grounded, node_lifted):
        if len(node_grounded[1:]) == 0:
            return 1
        return (len(node_lifted[1:]) - self._check_disj_pred(node_lifted, True)) / len(node_grounded[1:])

    def Omega(node_grounded, possible_matches):
        value = sum(omega(node_grounded, i) for i in possible_matches)
        if len(possible_matches) == 0:
            return 0
        return value / len(possible_matches)

    def alpha_factor(ori_not_recovered, gen_not_matched, vars_dict):
        A = ori_not_recovered
        if len(A) == 0:
            return 1, {}
        possible_matches = [every_match(i, gen_not_matched, vars_dict) for i in ori_not_recovered]
        value = sum(Omega(ori_not_recovered[i], possible_matches[i]) for i in range(len(ori_not_recovered)))
        return value / len(A), possible_matches

    def every_match(grounded, lifted_list, vars_restricted_domain):
        """
        Given a grounded node and a list of partially lifted nodes, tries to match the first with all of the second. Then, it returns the set of every
        possible match.
        :param grounded: Grounded landmark
        :param lifted_list: List of lifted landmarks
        :param vars_restricted_domain: negative domain of every variable
        :return: set of possible matches for the grounded landmark
        """
        every_match = set()
        for lifted_landmark in lifted_list:
            check_names = lifted_landmark.name == grounded.name
            check_types = False
            if check_names:
                check_types = all([i[1] == j[1] for i, j in zip(grounded.objs, lifted_landmark.objs)])
            if check_types and check_names:
                check = sum([
                    1 for p, q in zip(grounded.objs, lifted_landmark.objs)
                    if (p[0] == q[0] or ('?' in q[0] and p not in vars_restricted_domain[q])) and (p[1] == q[1])
                ])
                if check == len(grounded.objs):
                    every_match.add((lifted_landmark[0], lifted_landmark[1]))
        return every_match

    def __call__(self, node):
        required_again = set()
        state_facts = {i for i in node.state}
        # Check for accepted landmarks
        to_accept = self.landmarks - node.accepted
        # Dictionary to keep the alpha value of each pred
        for lm in to_accept:
            if self._check_in(lm, state_facts):
                pred_accepted = True
                if lm in self.lgg:
                    for pred in self.lgg[lm]:
                        if not pred in node.accepted:
                            pred_accepted = False
                            break
                if pred_accepted:
                    node.accepted.add(lm)
        # Check for required again landmarks
        yet_to_achieve = self.landmarks - node.accepted
        preds_to_achieve = set()
        for lm_p in yet_to_achieve:
            if lm_p in self.lgg:
                preds_to_achieve = preds_to_achieve.union(self.lgg[lm_p])
        for lm in node.accepted:
            # No tengo claro si es accepted_now o state_facts (si aceptamos un landmark disyuntivo y en un estado está una de sus posibles instanciaciones, pero no con
            # la que hemos aceptado el anterior, ¿qué ocurre?
            if not lm in state_facts:
                if lm in self.task.goals:
                    required_again.add(lm)
                else:
                    if self._check_in(lm, preds_to_achieve):
                        required_again.add(lm)
        return len((self.landmarks - node.accepted).union(required_again))


    #

    """def __call__(self, node):
        self.iter += 1
        if self.iter == 100:
            print("hihi")
        required_again = set()
        state_facts = {i for i in node.state}
        # Check for accepted landmarks
        to_accept = self.landmarks - node.accepted
        # Dictionary to keep the alpha value of each pred
        for lm in to_accept:
            if self._check_in(lm, state_facts):
                pred_accepted = True
                node.accepted.add(lm)

        return sum(self.costs[lm] * self.alpha_dict[lm] if lm in self.alpha_dict else self.costs[lm] for lm in
                   self.landmarks) - sum(
            self.costs[lm] * self.alpha_dict[lm] if lm in self.alpha_dict else self.costs[lm] for lm in node.accepted)"""

