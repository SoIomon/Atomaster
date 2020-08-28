# Atomaster
A bot for an interesting mobile game Atomas.
## Current progress & Method
1. Use Python script to control the game through ADB.
2. From the screen shot, the score and atoms' state are obtained by image recognition.
3. At present, the game information can be obtained on the NOX simulator.
    Screen shot and the result look like this:
    ![image](https://github.com/SoIomon/Atomaster/blob/master/Image/screenshot.png)
    ![image](https://github.com/SoIomon/Atomaster/blob/master/Image/result.png)
4. ...
## To-do list
1. Complete the game control algorithm: can start to play the game automatically.
## Development log
<font color=#2196F3 size=2 face="宋体">2020/08/28:</font></br>
1. Reduce the screenshot to speed up the update of atom list.
2. The method of shooting atom shoot_after_index() and catch atom catch_index() is added.
3. Add random play method random_ play()。