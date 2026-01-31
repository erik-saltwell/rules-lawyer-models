<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Rules Lawyer Models</h3>

  <p align="center">
    Pipeline for building various finetuned llm models
    <br />
    <a href="https://github.com/github_username/repo_name"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/github_username/repo_name">View Demo</a>
    &middot;
    <a href="https://github.com/github_username/repo_name/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/github_username/repo_name/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Here's a blank template to get started. To avoid retyping too much info, do a search and replace with your text editor for the following: `github_username`, `repo_name`, `twitter_handle`, `linkedin_username`, `email_client`, `email`, `project_title`, `project_description`, `project_license`

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```


### VS Code Dev Container Quickstart (after cloning)

This repo is set up to run inside a VS Code **Dev Container**, backed by **Docker Compose**. The container creates/uses a Python virtual environment at `/opt/venv` and runs `uv sync --frozen --inexact` automatically on first create, so you can start working immediately inside the container.

#### Prerequisites (one-time)

1. Install:

   * Docker Engine + Docker Compose
   * VS Code
   * VS Code extension: **Dev Containers**
2. If you want GPU acceleration locally:

   * Install NVIDIA drivers and ensure Docker can access the GPU (e.g., `nvidia-smi` works on the host).

> Note: This devcontainer configuration binds a 1Password SSH agent socket and SSH signing-related files from `.devcontainer/`. If you don’t use 1Password SSH agent or SSH commit signing, you may need to comment out or adjust those mounts in `.devcontainer/devcontainer.json` before opening in the container.

#### Open the project in the container

1. Clone and open the folder in VS Code:

   ```bash
   git clone <your-repo-url>
   cd rules-lawyer-models
   code .
   ```

2. In VS Code, open the Command Palette:

   * **Dev Containers: Reopen in Container**

VS Code will build the image (if needed), start the Compose service, mount your workspace into `/workspace`, and run the configured `postCreateCommand` (including `uv sync`).

#### Validate the environment (inside the container terminal)

Run these in the VS Code terminal (which should now be “inside” the container):

```bash
python -V
which python
uv --version
```

GPU check (optional):

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

Repo sanity check:

```bash
uv run pytest
```

(These checks match the intended “golden workflow” for this environment.)

#### Daily development loop

* Edit files normally in VS Code (they live on your host and are mounted into the container).
* Run code/tests in the container terminal:

  ```bash
  uv run pytest
  uv run python -m <your_module_here>
  ```

No rebuild is needed for normal code changes.

#### When you *do* need to rebuild

* If you changed dependencies (`pyproject.toml` / `uv.lock`) or anything in the container configuration:

  * VS Code Command Palette → **Dev Containers: Rebuild Container**
  * Or, from a host terminal:

    ```bash
    docker compose build
    ```

This setup is designed so dependency changes rebuild the relevant layers, while day-to-day code edits do not require rebuilding.

---


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- LICENSE -->
## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
