<h1 align="center"> 
  Solo Gym<br/>
  <img src="https://upload.wikimedia.org/wikipedia/en/1/1b/WPI_logo.png" 
    alt="WPI Logo" width=75px/> <br />
  <img src="https://github.com/WPI-MMR/gym-solo/workflows/Build/badge.svg" 
    alt="Build Status" />
  <a href='https://coveralls.io/github/WPI-MMR/gym_solo?branch=master'>
    <img src='https://coveralls.io/repos/github/WPI-MMR/gym_solo/badge.svg?branch=master' 
    alt='Coverage Status' /></a>
</h1>

<p align='center'><i>A custom open ai gym environment for Solo experimentation.
  </i>
  <br/>
  <br/>
  <img src="doc/pos-control-screenshot.png" alt="screenshot" width=600/>
</p>

---

**TL;DR**
- An [OpenAI Gym](https://gym.openai.com/)-conformant environment featuring 
  [ODRI's Solo 8](https://open-dynamic-robot-initiative.github.io/)
- OpenAI Gym API offers out-of-the box compatibility with most RL frameworks
- Fully encapsulated and multiprocessing-ready
- Easily extendable environments, rewards, and observations
- Automatically normalize observation and action spaces
- Realtime simulations supported for non-discrete timesteps

Note that this project is a heavy work-in-progress. While we may not be able
to work on your issue, we appreciate any feedback. Additionally, if you have
any interesting ideas that you'd like to implement, we'll happily take
contributions!

🌟 We appreciate your star, it helps!


# Installation
The recommended way to set up this environment is to use virtualenv's for its
sandboxing. With that:

1. Clone and navigate to this repository from your terminal
2. `python -m venv venv-gym-solo`
3. `source venv-gym-solo/bin/activate`
4. `pip install --upgrade wheel pip`
5. `pip install -e .`

And the package should be installed! Note that this is in development mode
so any local changes will be reflected in the package as well.


# Running Tests
All of our tests can be run by using python's built in `unittest`. However,due to how PyBullet works, we require some extra dependencies. To run the
tests, use the following procedure:

1. Install `xvfb` (on Ubuntu, you can do `sudo apt install xvfb`)
2. Follow the instructions above for creating the virtual environment,
   but use `pip install -e .[test]` instead of `pip install -e .`
4. Navigate to the root of the repository
5. `python -m unittest discover -v .`
