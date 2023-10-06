# ADP-SMC-UAV
A repository about ADP-based SMC controller for a UAV.
 
To satisfy the requirements of physical experiments, we use SMC to design both inner and outer-loop controllers.
However, we fix the inner-loop controller parameters and just use RL to train some hyperparameters of the outer-loop controller.

Therefore, the uav model, inner-loop controller, and outer-loop controller are integrated together as the "environment" of the RL.

And finally we use Rviz to display 3D UAV control performance. Hope we can get satisfactory performance as soon as possible.

One can download the package to the src directory in the workspace and run the SMC tests for UAV attitude and position by 
```commandline
$ catkin_make workspace_name # or catkin build
$ source devel/setup.bash
$ roslaunch adp_smc_uav_ros att_ctrl.launch # run UAV attitude control
$ roslaunch adp_smc_uav_ros pos_ctrl.launch # run UAV position control
```
