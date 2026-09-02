# What you need

Before you start, set up a few things on your computer. This is a one time job. Take it slow. Each item below has a link and a short reason.

## A GitHub account

GitHub is the website where your project lives. Think of it as a shared folder for code, with a history of every change.

- If you do not have an account, make one at [github.com](https://github.com).
- Tell your Novelcore contact your GitHub username. They will give your account access to the project. Without access you cannot see the project or send changes.

## Git

Git is the tool that copies the project to your computer and sends your changes back.

- Download it from [git-scm.com/downloads](https://git-scm.com/downloads).
- Install it with the default options. You do not need to change anything during setup.

!!! note "How do I know it worked?"
    Open your Terminal. On a Mac press `Cmd` and `Space`, type `Terminal`, press `Enter`. On Windows open the `Git Bash` app that was just installed. Type `git --version` and press `Enter`. If you see a version number, you are good.

## A code editor

You need something nicer than Notepad to open the project files. We suggest Visual Studio Code. It is free and friendly.

- Download it from [code.visualstudio.com](https://code.visualstudio.com).
- Install it with the default options.

This is where you will read files, edit them, and see the whole project as a tree on the left.

## Docker Desktop

Each step in your pipeline gets packed into a small box called an image. Docker is the tool that builds that box. You only need Docker installed so you can test a step on your own computer if you want to. The real builds happen on the platform, not on your laptop.

- Download it from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop).
- Install it and open it once so it finishes setting up.

!!! tip "You can skip Docker for now"
    You do not need Docker to add your first step and ship it. The platform builds every step for you. Install Docker later, only if you want to test a step on your own machine before you push.

## A quick checklist

Tick these off before moving on:

- [ ] I have a GitHub account.
- [ ] My Novelcore contact gave my account access to the project.
- [ ] Git is installed and `git --version` shows a number.
- [ ] Visual Studio Code is installed.
- [ ] Docker Desktop is installed (optional for now).

When these are done, go to [Get the project](get-the-project.md).
