from pandapower.control.basic_controller import BasicCtrl, Controller


class RLControllor(Controller):
    def __init__(self, net, in_service=True, order=0, level=0, index=None, recycle=False,
                 drop_same_existing_ctrl=False, initial_run=True, overwrite=False,
                 matching_params=None, **kwargs):
        super(RLControllor, self).__init__(net, index, **kwargs)
        self.new_action_set = False

    def set_new_action(self, action=None):
        """
        must be called after _get_state() and at start of _action()
               """
        assert self.new_action_set == False, "act"
        self.new_action_set = True

    def is_converged(self, net):
        """
        This method calculated whether or not the controller converged. This is
        where any target values are being calculated and compared to the actual
        measurements. Returns convergence of the controller.
        """

        return  not self.new_action_set

    def control_step(self, net):
        """
        If the is_converged method returns false, the control_step will be
        called. In other words: if the controller did not converge yet, this
        method should implement actions that promote convergence e.g. adapting
        actuating variables and writing them back to the data structure.
        """
        self.new_action_set = False
