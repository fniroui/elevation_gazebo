# elevation_gazebo
##### Python script to create worlds with uneven terrain for Gazebo simulator.
## How to use:
1. Clone the repository to your catkin_ws/src folder and build it.
2. To generate worlds, navigate to the ../script folder and run:
```
python map_gen -n 1 -s 10
```
  - Specify the number of worlds to generate (-n) and the size of the worlds in meters (-s).
3. To launch Gazebo run:
```
roslaunch elevation_gazebo environment.launch map:=1
```
  - Here the map number is the world number. The generated world files can be found under the ../worlds folder, and the generated elevation images can be found under ../worlds/models/maps folder.
