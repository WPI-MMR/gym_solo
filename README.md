<h1 align='center'> 
  Solo Gym<br/>
  <img src="https://upload.wikimedia.org/wikipedia/en/1/1b/WPI_logo.png" 
    alt="WPI Logo" width=75px style="padding:15px"/> <br />
  <img src="https://github.com/WPI-MMR/gym-solo/workflows/Build/badge.svg" 
    alt="Build Status" />
  <a href='https://coveralls.io/github/WPI-MMR/gym-solo?branch=main'>
    <img src='https://coveralls.io/repos/github/WPI-MMR/gym-solo/badge.svg?branch=main' 
    alt='Coverage Status' /></a>
</h1>

<p align='center'><i>A custom open ai gym environment for Solo experimentation.
  </i></p>

---

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