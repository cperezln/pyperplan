from .heuristic_base import Heuristic
from typing import IO, Any, Dict, Tuple, Optional, List, Union, Type, Sequence, cast

class LMCountHeuristic(Heuristic):

    def __init__(self, task, lgg, translations):
        self.task = task
        self.lgg = lgg
        self.translations = translations
        self.landmarks = set()
        for k, v in lgg.items():
            self.landmarks.add(k)
            for ele in v:
                self.landmarks.add(ele)

    def _check_disj_pred(self, pred):
        for obj in pred.replace(')','').replace('(','').split(' ')[1:]:
            if '?' in obj:
                return True
        return False

    def _check_in(self, pred, set):
        if not self._check_disj_pred(pred):
            return pred in set
        else:
            pred_as_list = pred.replace(')', '').replace('(', '').split(' ')
            new_facts = [[pred_as_list[0]]]
            for objs in pred_as_list[1:]:
                if objs in self.translations:
                    newer_facts = []
                    for new_fact in new_facts:
                        for pos_trad in self.translations[objs]:
                            newer_fact = [i for i in new_fact]
                            newer_fact += [pos_trad]
                            newer_facts.append(newer_fact)
                        new_facts.remove(new_fact)
                    new_facts += newer_facts
                else:
                    for new_fact in new_facts:
                        new_fact += [objs]
            newer_as_strings = ['(' + " ".join(i) + ')' for i in newer_facts]
            for new_pred in newer_as_strings:
                if new_pred in set:
                    return True
        return False

    def __call__(self, node):
        required_again = set()
        state_facts = {i for i in node.state}
        # Check for accepted landmarks
        to_accept = self.landmarks - node.accepted
        for lm in to_accept:
            if self._check_in(lm, state_facts):
                pred_accepted = True
                if lm in self.lgg:
                    for pred in self.lgg[lm]:
                        if not (pred in node.accepted):
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
                if self._check_in(lm, self.task.goals):
                    required_again.add(lm)
                else:
                    if lm in preds_to_achieve:
                        required_again.add(lm)
        return len((self.landmarks - node.accepted).union(required_again))