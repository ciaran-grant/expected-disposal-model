from dataclasses import dataclass

@dataclass
class ModellingDataContract:
    """ Holds details for defining modelling terms in a single place.
    """
    
    ID_COL = "Match_ID"
    RESPONSE = "Disposal_Response"
    TRAIN_TEST_SPLIT_COL = "ModellingFilter"
    
    action_types = ['Kick', 'Handball']
    outcome_types = ['effective', 'ineffective', 'clanger']
    feature_list = [
        'type_Kick_a0',
        'type_Handball_a0',
        'quarter_a0',
        'quarter_seconds_a0',
        'overall_seconds_a0',
        'left_right_start_x_a0',
        'left_right_start_y_a0',
        'left_right_end_x_a0',
        'left_right_end_y_a0',
        'dx_a0',
        'dy_a0',
        'movement_a0',
        'type_Kick_a1',
        'type_Handball_a1',
        'quarter_a1',
        'quarter_seconds_a1',
        'overall_seconds_a1',
        'left_right_start_x_a1',
        'left_right_start_y_a1',
        'left_right_end_x_a1',
        'left_right_end_y_a1',
        'dx_a1',
        'dy_a1',
        'movement_a1',
        'type_Kick_a2',
        'type_Handball_a2',
        'quarter_a2',
        'quarter_seconds_a2',
        'overall_seconds_a2',
        'left_right_start_x_a2',
        'left_right_start_y_a2',
        'left_right_end_x_a2',
        'left_right_end_y_a2',
        'dx_a2',
        'dy_a2',
        'movement_a2',
        # 'time_delta1',
        # 'time_delta2',
        # 'dx_a01',
        # 'dy_a01',
        # 'dx_a02',
        # 'dy_a02'
        ]
    
    monotone_constraints = {}
