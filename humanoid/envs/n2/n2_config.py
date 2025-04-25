from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class N2RoughCfg(LeggedRobotCfg):
    """
    Configuration class for the N2 humanoid robot.
    """
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.75]
        default_joint_angles = {
            "L_leg_hip_yaw_joint": 0.,
            "L_leg_hip_roll_joint": 0.,
            "L_leg_hip_pitch_joint": -0.1495,
            "L_leg_knee_joint": 0.3215,
            "L_leg_ankle_joint": -0.1720,
            "R_leg_hip_yaw_joint": 0.,
            "R_leg_hip_roll_joint": 0.,
            "R_leg_hip_pitch_joint": -0.1495,
            "R_leg_knee_joint": 0.3215,
            "R_leg_ankle_joint": -0.1720,
        }

    class env(LeggedRobotCfg.env):
        num_observations = 40
        num_privileged_obs = 43
        num_actions = 10

        enable_early_termination = True
        termination_height = 0.5
    
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_gains = True
        p_gain_range = [0.8, 1.2]
        d_gain_range = [0.8, 1.2]

        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]

        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]

        randomize_friction = True
        friction_range = [0.1, 2.]

        randomize_restitution = True
        restitution_range = [0., 1.]

        randomize_base_mass = True
        added_mass_range = [-5., 5.]

        dynamic_randomization = 0.02 
    
    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {
            'leg_hip_yaw_joint': 80.0, 'leg_hip_roll_joint': 80.0, 'leg_hip_pitch_joint': 120.0,
            'leg_knee_joint': 120.0, 'leg_ankle_joint': 15.0
        }
        damping = {
            'leg_hip_yaw_joint': 5.0, 'leg_hip_roll_joint': 5.0, 'leg_hip_pitch_joint': 5.0,
            'leg_knee_joint': 5.0, 'leg_ankle_joint': 1.0
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10 
    
    class sim(LeggedRobotCfg.sim):
        dt =  0.002

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/N2/urdf/N2_10dof.urdf'
        name = "Ning"
        foot_name = "ankle"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False
    
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False
    
    class rewards( LeggedRobotCfg.rewards ):
        cycle_time = 0.64
        soft_dof_pos_limit = 0.9
        base_height_target = 0.70
        max_contact_force = 300. # forces above this value are penalized
        target_feet_height = 0.05
        
        class scales( LeggedRobotCfg.rewards.scales ):
            # vel tracking
            tracking_lin_vel = 2.4
            tracking_ang_vel = 1.5
            # base pos
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = 1.0
            base_height = -10.0
            # style
            feet_air_time = 0.0
            default_joint_pos = 1.0
            feet_swing_height = -20.0
            # contact 
            contact = 1.5
            contact_no_vel = -0.2
            feet_contact_forces = -0.01
            # energy
            dof_acc = -2.5e-7
            energy_cost = -1e-3
            action_smoothness = -0.01
            # other
            collision = 0.0
            dof_pos_limits = -5.0

    class noise:
        add_noise = True
        noise_level = 1.0    # 1 scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.2
            lin_vel = 0.05
            gravity = 0.05
            quat = 0.05
            height_measurements = 0.1

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        min_cmd_vel = 0.3
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.

class N2RoughCfgPPO(LeggedRobotCfgPPO):
    class policy:
        class_name = "ActorCriticRecurrent"
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        class_name = "PPO"
        entropy_coef = 0.01

    class runner( LeggedRobotCfgPPO.runner ):
        max_iterations = 10000
        run_name = ''
        experiment_name = 'n2'
