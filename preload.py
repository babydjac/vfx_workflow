import os
import sys

# Define the requirements for the extension
REQUIRED_PACKAGES = [
    "gradio>=3.10.1",
    "transparent-background==1.3.2",
    "torch>=1.9.0",
    "Pillow",
]

ASCII_ART = """
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⡟⣯⡿⣝⣯⠿⣽⢯⣟⡿⣻⣟⣿⣿⣿⢿⣿⣿⣿⣿⣻⣿⣶⣾⣷⣬⡁⠈⢀⣀⣵⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡰⢌
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⢯⣽⢯⣟⢾⣵⣻⡽⣾⡽⣽⣳⢯⣟⣿⣿⣯⣿⢾⣿⣿⣟⡾⣯⢿⣿⣿⣿⣷⣤⣿⡇⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠠⠁⠊⠀⠁⠀⠀⠀⠀⠀⠂⠘⠂
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⣏⣿⢯⣿⢾⣟⡾⣳⣟⣳⣟⡷⣯⣟⣾⣿⣿⣿⢾⣿⣟⡾⣿⣿⣽⣿⣿⣏⣿⣿⣿⣿⣷⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⣿⡳⣞⣿⣿⡯⣿⢾⣽⣳⢯⡷⣯⣟⣷⣻⣽⣿⣿⣿⣿⡽⣯⣿⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⢀⣀⣄⡀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⢯⣗⣻⣿⣟⡾⣽⢿⣛⣾⡽⣯⢟⣷⣻⢾⣽⣿⣿⣞⣯⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⠀⠀⠀⠀⠀⠀⠀⣀⣀⣠⣶⠿⣻⠿⠛⢹⣿⢿⠟⠛⠉⠐⣌⢻⡀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡿⣝⣾⣿⡟⣼⣻⣏⣿⣹⢺⣟⣵⣫⣶⡿⣾⣿⣿⣷⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡄⢀⣠⣴⡾⢋⣽⣿⣯⣿⠋⡔⠉⣼⣿⠇⠀⠀⠀⠣⢌⣻⠇⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡿⣽⣿⣟⢾⣿⡵⣯⢾⣱⣯⣿⣷⢯⣷⣿⣿⣿⣿⣿⣿⣿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣿⣿⣿⣿⣿⣿⣷⣾⣿⣿⣿⣴⣿⣿⣿⣿⠃⢌⠀⢰⣿⡏⠀⠀⠀⠀⠱⢨⠼⡇⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⣻⣿⣿⣯⢿⣿⣟⣯⣿⣯⣿⣿⣿⣯⣿⣟⣿⣿⣳⣿⣿⣿⣈⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢳⣿⢣⠘⣀⡞⢸⣿⠁⠀⠀⠀⠀⢀⠃⢾⠁⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⣻⣽⣿⣿⣯⢿⣿⣻⣾⢿⣿⣽⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣭⣴⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣟⣧⣿⢋⢆⠃⣾⠀⣸⡟⠀⠀⠀⠀⠀⠀⠎⡽⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⣾⡟⣽⣿⣿⣿⣯⣿⣿⣏⣿⣿⢿⣿⣧⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⡟⢯⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣿⡿⢿⡿⢣⢍⣢⣭⡇⢀⣿⢃⠀⠀⠀⠀⠀⢈⢒⡇⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⣼⡟⢰⣿⢾⣿⣯⣷⣿⣿⣿⣞⣿⣿⣿⣿⣦⣛⢿⣿⣿⣿⣿⣿⣿⣽⠏⠉⠗⣚⣩⣾⠟⣭⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣗⣿⣿⣿⣿⣷⣿⡏⢠⠃⢸⡟⠰⠀⠀⠀⠀⠀⢌⣺⠁⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢰⡿⠀⣾⡏⣿⣿⡷⣿⣿⣿⣿⣧⣧⣼⣿⣿⣿⣿⣶⡹⢿⣿⣿⣿⣿⡸⢄⠉⠉⡙⣉⠡⣚⣵⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⣿⣿⣇⡞⠀⣼⠁⠁⠀⠀⠀⠀⠈⡴⡇⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⣿⠁⢸⡟⠀⣿⣯⣿⣿⣿⣿⣿⣿⣿⣿⣟⠻⢾⣫⣼⠗⠀⠙⢟⡻⣿⣷⡂⠀⠀⠐⠀⠂⢡⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⣿⣿⣿⠃⣿⣿⠟⠀⢠⡇⠀⠀⠀⠀⠀⠀⣘⡾⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢸⡇⢀⣿⠁⠀⣿⣿⣳⣿⣿⣿⣿⣿⣿⣿⣬⠉⡒⠋⠁⠀⠀⠀⣤⠙⠌⠻⢆⡀⠀⠀⢀⣰⡿⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢸⣿⠋⠀⠀⣸⠁⠀⠀⠀⠀⠀⠐⣼⠁⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢸⢀⣾⠏⠀⢰⡿⣿⣳⣿⣿⣿⣿⣿⣿⣿⣿⣷⣌⢡⣁⡀⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠤⠚⢱⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⣸⠇⠀⠀⢠⠇⠀⠀⠀⠀⠀⢀⣹⡏⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣿⣾⠟⠀⢀⣾⣟⣿⣟⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡻⢎⣀⣀⠄⠀⠀⣤⣠⣤⡀⠀⠀⠀⣠⡿⣿⣿⣿⣿⣿⡟⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢟⠏⠀⠀⢠⠏⠀⠀⠀⠀⠀⠠⣜⡾⣷⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢠⣟⡞⠀⠀⣼⣿⣞⡿⣾⣿⣾⢿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣄⣀⠀⠀⠀⠀⠉⠉⠀⠀⢀⡼⢏⣵⣿⣿⣿⡟⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠣⠋⠀⢀⣴⠏⠀⡄⠀⠀⠀⣀⢳⡞⠀⢿⡀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⣴⣟⣷⣇⢀⣼⠏⢸⢾⡿⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⣿⡲⢤⣄⣀⣠⠖⣏⣱⣾⣿⣿⣿⣿⠇⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠉⠓⢶⣾⠟⠁⠀⡜⠀⠀⠀⡰⣬⠟⠀⠀⠘⣧⠀⠀⠀⠀⠀⠀
⠀⢠⣾⡳⠋⢸⢾⡾⡟⠀⣾⣿⣟⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢷⣃⢎⠲⣁⠏⣴⣿⡿⢻⢽⣿⣿⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⢃⠈⣆⠀⠀⠀⣠⠞⠀⠀⢀⡰⣳⠋⠀⠀⠀⠀⠘⠆⠀⠀⠀⠀⠀
⣰⣻⠞⠀⢀⣼⣿⣿⠀⢰⣟⣾⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢋⡜⣻⣆⠣⠜⡒⢼⡿⣈⠗⣚⣏⠿⣼⣿⡿⣿⣿⣿⢿⣙⢮⡿⠃⠀⠈⠒⠶⠾⠋⠀⠀⢀⢦⡟⠁⠀⠀⠀⠀⠀⠀⠈⢆⠀⠀⠀⠀
⡗⠁⠀⣠⡾⢃⣽⡾⣧⡾⣽⣯⣿⣿⣿⣿⣿⡿⠻⠿⠿⣿⣿⣿⣿⣿⣿⣯⣇⠎⡴⢡⢎⠳⡜⢺⣄⢛⣦⢙⡴⢪⠜⣆⣛⣷⣿⣿⣟⢣⠎⠻⠁⠀⠀⠀⠀⠀⠀⠀⠀⡄⣯⡞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣧⠀⠀⠀
⠀⢀⣾⢋⣴⠟⠁⠻⣽⡿⣿⣾⣿⣿⡟⠉⠉⢿⣄⠀⡸⠛⢉⠛⣿⣿⣿⣷⣮⣛⠍⠲⢌⠣⡜⡄⠛⡷⢏⠲⡌⣧⣻⠞⣋⢶⣿⣿⣏⠆⠁⠀⠀⠀⠀⠀⠀⠀⠀⡄⣳⣾⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⡄⠀⠀
⣴⣿⣵⠟⠁⠀⢀⣼⣿⣿⢷⣿⣿⡟⠀⠀⠀⠀⠙⣏⠀⠁⢢⡀⠸⣿⣿⣿⣿⠿⣿⣷⡾⠶⢶⣌⢓⡰⢊⣕⡾⢛⠤⡙⡔⢺⣿⢿⣿⠀⠀⠀⠀⠀⠀⠀⠀⡄⣣⣼⡏⢽⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣧⠀⠀
⠟⢏⠀⠀⠀⢀⣾⣿⣟⣿⣿⣿⣿⡁⠀⠀⠀⠀⠀⠸⡄⠀⠀⠈⠂⣿⣿⣿⣿⣿⣷⣤⡙⣷⣬⡘⠳⠐⢃⠌⡰⢉⠆⡱⢈⣽⣿⡎⢿⡆⠰⣇⠀⡀⢄⠢⣍⣶⣿⣿⡅⢺⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⡿⠀⠀
⠀⠀⠉⠒⢀⡾⠛⣿⣿⣿⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⢱⠀⠀⠀⢠⣿⣿⣿⣇⢻⣿⣿⢿⣾⣿⣷⠀⠁⠂⠌⢀⠁⠂⠁⢰⣿⣿⣷⡈⢻⡄⢹⡖⣈⠦⡿⣾⣿⣿⣿⣿⢸⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣟⠀⠀
⡰⢪⡕⡴⠃⢠⣾⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠈⢿⣄⡀⣼⣿⠟⣹⠇⠀⢿⣿⠀⣿⣿⣧⡀⠀⠀⠀⠀⠀⠀⠀⠘⠆⣻⣿⣿⣦⣽⣾⣧⣝⣾⣷⢻⣿⣿⣿⣿⣷⣿⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⠇⠀⠀
⢡⣣⠞⠀⣰⣿⣿⢋⣿⣿⣿⣿⡷⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣷⡿⠟⠊⠁⠀⣠⣿⣏⣾⣿⣾⡿⢿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⢿⣿⣿⡞⣿⣿⣿⣿⣿⣿⣿⡀⠀⠀⠀⠀⠀⠀⠀⣠⣾⡿⠀⠀⠀
⠚⠁⠀⢀⣿⣷⠏⣼⣿⣿⣿⣿⣟⠀⠀⠀⠀⠀⠀⢀⣠⣶⢿⡟⠀⠀⢀⠄⢊⣿⣿⣿⣿⠟⠛⠉⠁⠀⠀⠰⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠛⢿⣿⣿⣿⣿⢿⣿⣿⣷⡀⠀⠀⠀⠀⠒⠛⠛⠁⠀⠀⠀⠀
⠀⠀⢀⣼⣿⣟⣴⣿⣿⣿⣿⣿⡏⠀⠀⠀⣀⣤⣾⣿⠟⣉⢾⠇⠀⠐⣁⢶⣿⣿⠟⠋⠀⠀⠀⠀⠀⠀⠀⠀⠙⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠻⣿⡆⠈⢻⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⣠⣾⣟⣾⣿⣿⣿⣿⣿⣿⣿⠇⢀⣤⠾⠛⠋⠁⠀⠈⡔⣾⠖⢀⠜⣠⡾⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢙⢄⡀⠻⣿⠙⠳⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀
⡴⣟⡷⣯⣿⢿⣿⣿⡿⣿⣿⣿⡴⠋⠀⠀⠀⠀⠀⠀⠐⣸⡿⠀⠘⡶⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠲⢿⣷⡀⠀⠻⡷⣄⠀⠀⠀⠀⠀⠀
⡸⠋⠉⠙⢣⣿⣿⣿⣇⣿⣿⠏⠀⠀⠀⠀⠀⠀⠀⠀⢰⡿⠁⠀⢰⣁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⡝⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡖⣦⡘⣧⡀⠀⠉⢌⠳⡀⠀⠀⠀⠀
⠀⠀⠀⠀⢸⣿⢾⣿⣿⣿⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠁⠀⠀⠘⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠨⣿⠌⢧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠓⡛⠁⢣⠀⠀⠀⠡⡘⢆⠀⠀⠀
⣦⣀⣀⣤⢾⣿⣻⣿⣿⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⢸⠇⠀⠀⠀⠀⠀⠀⠀⠀⣠⢴⣲⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢽⡚⡌⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⠘⢸⠀⠀⠀⠀⠐⣌⡆⠀⠀
⢿⡟⠏⠻⢿⣷⣻⣿⣿⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠻⠟⠗⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢺⡝⡔⠁⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠀⣼⡇⠀⠀⠀⠀⢸⡷⠀⠀
⡏⠀⠀⠀⠀⠙⢿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠠⢁⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢾⡹⢬⡁⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⣿⡇⠀⠀⠀⢀⣾⣿⠀⠀
⢇⡀⠂⠀⠀⠀⠈⠙⠿⣯⠀⠀⠀⠀⠀⠀⢀⠑⡂⣯⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⡘⣽⡝⢦⡃⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣺⣿⣿⣷⣄⠀⢀⣾⣿⡇⠀⠀
⡄⠀⠀⠀⠀⠀⠀⠀⠀⠈⠑⣄⠀⠀⠀⠀⠠⢃⡕⢺⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⡑⢼⡟⡼⣩⠿⣦⡡⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠰⣌⣷⣿⣿⣿⣿⣿⣶⣿⣿⣿⠁⠀⠀
⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⢐⡈⢳⡄⠀⢠⠡⢣⠜⣹⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢄⠣⣹⢾⡙⠶⡡⢞⡩⢷⣯⡰⠡⢄⠠⣀⠀⡄⢠⢂⡜⣬⣷⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇⠀⠀⠀
⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⡘⢄⢻⡄⢣⠜⣡⢞⣿⣿⣷⡆⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠘⢤⡷⢏⢧⡙⣣⠕⡪⠜⡡⢞⡹⠿⣮⣵⣦⣹⣬⣷⣾⣾⡙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀
⣿⣿⣧⡀⠀⠀⠀⠀⠀⠀⠀⠐⡈⢦⡙⢦⡙⢦⣻⢡⣿⣿⣿⣦⣐⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠰⣈⣦⢽⢫⠜⡩⢆⡱⢢⠙⠄⠃⠁⠊⠔⢫⠔⢦⠣⣝⢲⣿⠿⣿⡅⢹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡀⠀
⣿⣿⣿⣇⢤⠀⠀⠀⠀⠀⠀⠀⢈⠒⡜⢦⣹⣳⣿⣿⣿⣿⣿⡿⢠⣷⠶⢤⣤⣀⣀⣀⣄⣠⣤⣬⠶⢓⢋⠆⢣⠊⠌⠑⡀⠂⠁⠈⠀⠀⠀⠀⠈⠐⣊⠦⡙⢦⣿⣿⣨⣿⣤⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡀
⣿⣿⣿⣿⣿⣶⢄⡀⠀⠀⠀⠀⠠⢩⢜⣣⢷⢃⣿⣿⣿⣿⣿⣇⣿⣿⣧⠀⠀⠈⠉⠉⠈⠁⠀⠀⠐⠈⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⠠⢎⣹⣿⣿⢿⣿⣿⢿⣿⣿⣿⣿⣿⣿⠙⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣧⣿⣦⣿⣥⣤⣀⣀⠂⡇⣞⣼⣿⣿⣿⣿⣿⢿⣿⣿⣿⡿⠛⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⣨⡿⢻⣿⣿⠏⠀⣸⣿⣿⣿⣿⣿⡿⠀⣿⢿⣿⣿⣿⣿⣿⣿
⡇⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⠙⣾⣾⣿⣯⣿⣿⣿⣏⣼⡿⠛⠙⠁⠀⢹⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠱⣿⣿⣿⡿⣏⠀⣰⣿⣿⣿⣿⣿⣿⡇⠀⡿⢸⣿⣿⣿⣿⣿⣟
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⢰⣿⡿⠟⠛⢫⡹⠖⠉⠁⠀⠀⠀⠀⠀⠘⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⢻⣿⡿⢻⡇⠉⣹⣿⣿⣿⡟⣿⣿⡿⢀⣼⣡⣿⣿⣿⣿⣿⣿⣯
⡿⢿⠟⣿⣿⣿⣿⣿⣿⣿⡿⠟⢋⣁⡤⠒⣠⠞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢲⣿⣿⡀⠘⢛⣿⣿⣿⣿⡟⢀⣿⣿⣅⣾⣿⣿⣿⣿⣿⣿⣿⣿⣧
⣿⣿⢰⣿⣿⣿⣿⣿⠛⢣⠐⢌⣖⠡⠔⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡰⠂⠀⠀⠀⠀⠀⠀⠀⠀⢨⡟⠙⢿⡿⠿⣻⣿⣿⣿⠟⣠⣿⡿⣻⣿⣿⢟⣋⣱⣿⣿⣿⣿⣿⣿
⣿⢿⢿⣿⣭⡿⠃⠄⠀⡠⠈⠸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠔⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⠀⠀⠀⣹⣾⣿⣿⣿⣿⣾⠿⠋⠀⠹⡸⣏⡉⠉⣿⣿⣿⣿⣿⣿⠋
⡟⠁⢸⡾⠋⠀⢡⡤⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡾⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡾⢽⠂⢀⣼⣿⠟⠉⠉⠋⠉⠀⠀⠀⣆⣠⠇⠀⠀⣼⣿⣿⣿⣿⣿⣧⡀
⣇⠀⠋⠀⠀⠀⡺⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡃⢻⡇⢸⣿⢻⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⣿⣿⣿⣿⡿⠏⠉⠉
⣿⡀⠀⠀⠀⠠⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠂⠌⡑⣇⠘⣿⠈⢆⡀⠀⠀⠀⣀⠀⠀⣀⠤⣊⡽⢟⠿⠛⠉⠁⠐⠠⠀⠀
"""

def check_requirements():
    """Checks if all required packages are installed."""
    missing_packages = []
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package.split("==")[0].split(">=")[0].strip())
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"\033[91m[!] Missing Required Packages:\033[0m {', '.join(missing_packages)}")
        print("\033[93m[!] Please run the install.py script to resolve dependencies.\033[0m")
        return False
    return True

def colorful_welcome():
    """Displays a colorful welcome message for the extension."""
    print("\033[95m" + "=" * 60)
    print(" " * 12 + "✨ Welcome to the VFX Workflow Extension! ✨")
    print("=" * 60 + "\033[0m")
    print(ASCII_ART)
    print("\033[94mThis extension provides a seamless video-to-frame processing workflow.\033[0m")
    print("\033[92mFeatures include:\033[0m")
    print("  ➡ Frame extraction with precise FPS handling.")
    print("  ➡ Mask creation using advanced AI techniques.")
    print("  ➡ Keyframe management with dynamic subfolder splitting.")
    print("  ➡ Easy integration with Img2Img workflows.")
    print("  ➡ Automatic generation of ZIP files and EBSynth configurations.\n")
    print("\033[94mMake sure you have all required packages installed!\033[0m")

if __name__ == "__main__":
    colorful_welcome()
    if not check_requirements():
        sys.exit("\033[91m[!] Please install missing dependencies using install.py.\033[0m")
