# Docker 部署教程 (freegoogleapi)

本文档提供了使用 Docker 构建和运行 `freegoogleapi` 项目的详细步骤。

**先决条件:**
*   确保您的系统已正确安装并正在运行 Docker。您可以从 [Docker 官方网站](https://www.docker.com/get-started) 下载并安装 Docker Desktop (适用于 Windows 和 macOS) 或 Docker Engine (适用于 Linux)。
*   项目代码已下载到本地。

## 1. 理解项目中的 Docker 相关文件

在项目根目录下，您会找到以下与 Docker 配置相关的文件：

*   **[`Dockerfile`](./Dockerfile:1):** 这是构建 Docker 镜像的蓝图。它定义了基础镜像、依赖项安装、代码复制、端口暴露以及容器启动时执行的命令。
*   **[`.dockerignore`](./.dockerignore:1):** 这个文件列出了在构建 Docker 镜像时应忽略的文件和目录。这有助于减小镜像大小并加快构建速度，例如排除 `.git` 目录、本地开发环境文件等。
*   **[`supervisord.conf`](./supervisord.conf:1):** (如果项目使用 Supervisor) Supervisor 是一个进程控制系统，它允许用户在类 UNIX 操作系统上监控和控制多个进程。此配置文件定义了 Supervisor 应如何管理应用程序的进程 (例如，主服务和流服务)。

## 2. 构建 Docker 镜像

要构建 Docker 镜像，请在项目根目录下打开终端或命令行界面，然后执行以下命令：

```bash
docker build -t freegoogleapi:latest .
```

**命令解释:**

*   `docker build`: 这是 Docker CLI 中用于构建镜像的命令。
*   `-t freegoogleapi:latest`: `-t` 参数用于为镜像指定一个名称和可选的标签 (tag)，格式为 `name:tag`。
    *   `freegoogleapi`: 是您为镜像选择的名称。
    *   `latest`: 是标签，通常表示这是该镜像的最新版本。您可以根据版本控制策略选择其他标签，例如 `freegoogleapi:1.0`。
*   `.`: (末尾的点号) 指定了 Docker 构建上下文的路径。构建上下文是指包含 [`Dockerfile`](./Dockerfile:1) 以及构建镜像所需的所有其他文件和目录的本地文件系统路径。点号表示当前目录。Docker 守护进程会访问此路径下的文件来执行构建。

构建过程可能需要一些时间，具体取决于您的网络速度和项目依赖项的多少。成功构建后，您可以使用 `docker images` 命令查看本地已有的镜像列表，其中应包含 `freegoogleapi:latest`。

## 3. 运行 Docker 容器

镜像构建完成后，您可以使用以下命令来创建并运行一个基于该镜像的 Docker 容器：

```bash
docker run -d \
    -p <宿主机_服务端口>:2048 \
    -p <宿主机_流端口>:3120 \
    -v "$(pwd)/auth_profiles":/app/auth_profiles \
    # 可选: 如果您想使用自己的 SSL/TLS 证书，请取消下面一行的注释。
    # 请确保宿主机上的 'certs/' 目录存在，并且其中包含应用程序所需的证书文件。
    # -v "$(pwd)/certs":/app/certs \
    -e SERVER_PORT=2048 \
    -e STREAM_PORT=3120 \
    # 可选: 如果您需要设置内部 Camoufox 代理，请取消下面一行的注释，
    # 并将 "http://your_proxy_address:port" 替换为您的代理实际地址和端口。
    # -e INTERNAL_CAMOUFOX_PROXY="http://your_proxy_address:port" \
    --name freegoogleapi_container \
    freegoogleapi:latest
```

**命令解释:**

*   `docker run`: 这是 Docker CLI 中用于从镜像创建并启动容器的命令。
*   `-d`: 以“分离模式”(detached mode) 运行容器。这意味着容器将在后台运行，您的终端提示符将立即可用，而不会被容器的日志输出占用。
*   `-p <宿主机_服务端口>:2048`: 端口映射 (Port mapping)。
    *   此参数将宿主机的某个端口映射到容器内部的 `2048` 端口。`2048` 是应用程序主服务在容器内监听的端口。
    *   您需要将 `<宿主机_服务端口>` 替换为您希望在宿主机上用于访问此服务的实际端口号 (例如，如果您想通过宿主机的 `8080` 端口访问服务，则使用 `-p 8080:2048`)。
*   `-p <宿主机_流端口>:3120`: 类似地，此参数将宿主机的某个端口映射到容器内部的 `3120` 端口，这是应用程序流服务在容器内监听的端口。
    *   您需要将 `<宿主机_流端口>` 替换为您希望在宿主机上用于访问流服务的实际端口号 (例如 `-p 8081:3120`)。
*   `-v "$(pwd)/auth_profiles":/app/auth_profiles`: 卷挂载 (Volume mounting)。
    *   此参数将宿主机当前工作目录 (`$(pwd)`) 下的 `auth_profiles/` 目录挂载到容器内的 `/app/auth_profiles/` 目录。
    *   这样做的好处是：
        *   **持久化数据:** 即使容器被删除，`auth_profiles/` 中的数据仍保留在宿主机上。
        *   **方便配置:** 您可以直接在宿主机上修改 `auth_profiles/` 中的文件，更改会实时反映到容器中 (取决于应用程序如何读取这些文件)。
    *   **重要:** 在运行命令前，请确保宿主机上的 `auth_profiles/` 目录已存在。如果应用程序期望在此目录中找到特定的配置文件，请提前准备好。
*   `# -v "$(pwd)/certs":/app/certs` (可选，已注释): 挂载自定义证书。
    *   如果您希望应用程序使用您自己的 SSL/TLS 证书而不是自动生成的证书，可以取消此行的注释。
    *   它会将宿主机当前工作目录下的 `certs/` 目录挂载到容器内的 `/app/certs/` 目录。
    *   **重要:** 如果启用此选项，请确保宿主机上的 `certs/` 目录存在，并且其中包含应用程序所需的证书文件 (通常是 `server.crt` 和 `server.key` 或类似名称的文件)。应用程序也需要被配置为从 `/app/certs/` 读取这些证书。
*   `-e SERVER_PORT=2048`: 设置环境变量。
    *   `-e` 参数用于在容器内设置环境变量。
    *   这里，我们将 `SERVER_PORT` 环境变量设置为 `2048`。应用程序在容器内会读取此变量来确定其主服务应监听哪个端口。这应与 [`Dockerfile`](./Dockerfile:1) 中 `EXPOSE` 指令以及 [`supervisord.conf`](./supervisord.conf:1) (如果使用) 中的配置相匹配。
*   `-e STREAM_PORT=3120`: 类似地，设置 `STREAM_PORT` 环境变量为 `3120`，供应用程序的流服务使用。
*   `# -e INTERNAL_CAMOUFOX_PROXY="http://your_proxy_address:port"` (可选，已注释): 设置内部 Camoufox 代理。
    *   如果您的应用程序需要通过一个特定的内部代理服务器来访问 Camoufox 或其他外部服务，可以取消此行的注释，并将 `"http://your_proxy_address:port"` 替换为实际的代理服务器地址和端口 (例如 `http://10.0.0.5:7890` 或 `socks5://proxy-user:proxy-pass@10.0.0.10:1080`)。
*   `--name freegoogleapi_container`: 为正在运行的容器指定一个名称。
    *   这使得管理容器更加方便。例如，您可以使用 `docker stop freegoogleapi_container` 来停止这个容器，或使用 `docker logs freegoogleapi_container` 来查看其日志。
    *   如果您不指定名称，Docker 会自动为容器生成一个随机名称。
*   `freegoogleapi:latest`: 指定要运行的镜像的名称和标签。这必须与您在 `docker build` 命令中使用的名称和标签相匹配。

**首次运行前的重要准备:**
*   **创建 `auth_profiles/` 目录:** 在项目根目录下 (与 [`Dockerfile`](./Dockerfile:1) 同级)，手动创建一个名为 `auth_profiles` 的目录。如果您的应用程序需要初始的认证配置文件，请将它们放入此目录中。
*   **(可选) 创建 `certs/` 目录:** 如果您计划使用自己的证书并取消了相关卷挂载行的注释，请在项目根目录下创建一个名为 `certs` 的目录，并将您的证书文件 (例如 `server.crt`, `server.key`) 放入其中。

## 4. 管理正在运行的容器

一旦容器启动，您可以使用以下 Docker 命令来管理它：

*   **查看正在运行的容器:**
    ```bash
    docker ps
    ```
    (如果您想查看所有容器，包括已停止的，请使用 `docker ps -a`)

*   **查看容器日志:**
    ```bash
    docker logs freegoogleapi_container
    ```
    (如果您想持续跟踪日志输出，可以使用 `-f` 参数: `docker logs -f freegoogleapi_container`)

*   **停止容器:**
    ```bash
    docker stop freegoogleapi_container
    ```

*   **启动已停止的容器:**
    ```bash
    docker start freegoogleapi_container
    ```

*   **重启容器:**
    ```bash
    docker restart freegoogleapi_container
    ```

*   **进入容器内部 (获取一个交互式 shell):**
    ```bash
    docker exec -it freegoogleapi_container /bin/bash
    ```
    (或者 `/bin/sh`，取决于容器基础镜像中可用的 shell。这对于调试非常有用。)

*   **删除容器:**
    首先需要停止容器，然后才能删除它。
    ```bash
    docker stop freegoogleapi_container
    docker rm freegoogleapi_container
    ```
    (如果您想强制删除正在运行的容器，可以使用 `docker rm -f freegoogleapi_container`，但不建议这样做，除非您知道自己在做什么。)

## 5. 更新应用程序

当您更新了应用程序代码并希望部署新版本时，通常需要执行以下步骤：

1.  **停止并删除旧的容器** (如果它正在使用相同的端口或名称)：
    ```bash
    docker stop freegoogleapi_container
    docker rm freegoogleapi_container
    ```
2.  **重新构建 Docker 镜像** (确保您在包含最新代码和 [`Dockerfile`](./Dockerfile:1) 的目录中)：
    ```bash
    docker build -t freegoogleapi:latest .
    ```
3.  **使用新的镜像运行新的容器** (使用与之前相同的 `docker run` 命令，或根据需要进行调整)：
    ```bash
    docker run -d \
        -p <宿主机_服务端口>:2048 \
        # ... (其他参数与之前相同) ...
        --name freegoogleapi_container \
        freegoogleapi:latest
    ```

## 6. 清理

*   **删除指定的 Docker 镜像:**
    ```bash
    docker rmi freegoogleapi:latest
    ```
    (注意：如果存在基于此镜像的容器，您需要先删除这些容器。)

*   **删除所有未使用的 (悬空) 镜像、容器、网络和卷:**
    ```bash
    docker system prune
    ```
    (如果想删除所有未使用的镜像，不仅仅是悬空的，可以使用 `docker system prune -a`)
    **警告:** `prune` 命令会删除数据，请谨慎使用。

希望本教程能帮助您成功地通过 Docker 部署和运行 `freegoogleapi` 项目！