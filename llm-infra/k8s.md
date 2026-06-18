# K8s

![Image](../.gitbook/assets/authcode)

## 0 阅读路径

| 阶段   | 你会做什么                        | 核心对象                                |
| ---- | ---------------------------- | ----------------------------------- |
| 理解概念 | 知道 K8s 为什么存在，以及它和 Docker 的边界 | Cluster、Node、Pod、Deployment、Service |
| 启动环境 | 启动本地集群，配置 kubectl 和命名空间      | Context、Namespace                   |
| 部署应用 | 创建 Deployment，扩容，观察自愈和滚动更新   | Deployment、ReplicaSet、Pod           |
| 访问应用 | 给 Pod 提供稳定入口，理解 Service 类型   | Service、Endpoint、Ingress            |
| 接近生产 | 补配置、密钥、健康检查、资源限制和自动扩缩容       | ConfigMap、Secret、Probe、HPA          |

***

## 1 概念

**K8s 是 Kubernetes 的简称**。Kubernetes 这个单词中间有 8 个字母，所以写成 `K + 8 + s`。日常沟通里可以直接读 Kubernetes，也常读作 K-eights。

它是一个用来**管理容器化应用的平台**。如果你的应用已经被打包成 Docker 镜像，K8s 可以帮你部署、调度、扩容、重启、滚动发布，并把流量转发到健康的副本上。

**Docker 负责什么**

* 把应用和依赖打包成镜像
* 在单机上启动容器
* 提供容器运行时能力

**K8s 负责什么**

* 把大量容器调度到不同节点
* 维持副本数和健康状态
* 管理发布、回滚、配置、网络和扩缩容

生产环境里，一个后端服务通常不只是 `docker run my-api`。你还需要多个副本、失败自动重建、流量分发、配置管理、密钥管理、发布回滚和资源限制。K8s 的核心价值，就是把这些运维动作声明成资源对象，由控制器持续维护。

| 概念             | 含义                           | 一句话理解                        |
| -------------- | ---------------------------- | ---------------------------- |
| **Cluster**    | 一个 Kubernetes 集群，由控制面和工作节点组成 | 整套 K8s 系统                    |
| **Node**       | 集群里的机器，可以是虚拟机或物理机            | 真正运行 Pod 的机器                 |
| **Pod**        | K8s 里最小的可部署计算单元，通常包含一个容器     | 容器的运行外壳                      |
| **Deployment** | 声明 Pod 模板、副本数和发布策略           | 管理无状态应用                      |
| **ReplicaSet** | 保证符合 selector 的 Pod 数量       | Deployment 背后的副本控制器          |
| **Service**    | 给一组 Pod 提供稳定访问入口             | Pod 会变，Service 名字和虚拟 IP 相对稳定 |
| **Ingress**    | 管理外部 HTTP/HTTPS 路由           | 把域名和路径路由到 Service            |
| **ConfigMap**  | 存放非敏感配置                      | 配置不要写死进镜像                    |
| **Secret**     | 存放密码、Token、证书等敏感信息           | 敏感值不要明文写进 YAML               |
| **Namespace**  | 隔离同一集群内的资源                   | 练习、测试、生产可以分开管理               |

![Image](<../.gitbook/assets/authcode (1)>)

### 1.1 期望状态和控制循环

Kubernetes 的工作方式不是“执行一条命令然后结束”，而是让你声明一个**期望状态**，再由控制器不断比较实际状态和期望状态，发现偏差就修正。

理解这条控制循环后，Deployment 自动补 Pod、Service 自动找 Pod、HPA 自动改副本数都会变得直观：它们本质上都是控制器在让实际状态追上期望状态。

***

## 2 本地启动

本文推荐用 OrbStack 的内置 Kubernetes 做本地练习。Minikube、kind、Docker Desktop Kubernetes 也可以，命令思路基本一致。

### 2.1 启动集群

```bash
# 第一次启动会拉镜像，可能需要等 1 到 2 分钟
orb start k8s

# 确认 kubectl 当前上下文
kubectl config current-context

# 确认节点状态
kubectl get nodes
```

如果一切正常，你会看到 1 个节点，状态是 `Ready`。

### 2.2 安装和理解 kubectl

`kubectl` 是你和 Kubernetes API Server 交互的命令行工具。常见命令格式是：

```
kubectl <动作> <资源类型> <资源名> [参数]
```

| 命令                                | 作用                |
| --------------------------------- | ----------------- |
| `kubectl get pods`                | 查看 Pod 列表         |
| `kubectl describe pod <pod-name>` | 查看 Pod 详情、事件和调度原因 |
| `kubectl logs <pod-name>`         | 查看容器日志            |
| `kubectl apply -f app.yaml`       | 按 YAML 声明创建或更新资源  |
| `kubectl delete -f app.yaml`      | 删除 YAML 中定义的资源    |

如果提示找不到 `kubectl`，可以安装：

```bash
brew install kubectl
```

### 2.3 创建练习命名空间

```bash
kubectl create namespace lab
kubectl config set-context --current --namespace=lab
```

后续命令默认都作用在 `lab`。这样练习资源和系统命名空间隔离，最后也方便一键清理。

### 2.4 排障顺序

1. 先看资源是否存在：`kubectl get pods,deploy,svc`
2. 再看 Pod 状态和事件：`kubectl describe pod <pod-name>`
3. 然后看当前日志：`kubectl logs <pod-name>`
4. 如果容器反复崩溃，看上一次日志：`kubectl logs <pod-name> --previous`
5. 最后看命名空间事件：`kubectl get events --sort-by=.metadata.creationTimestamp`

***

## 3 部署应用：Deployment、ReplicaSet、Pod、Container

Pod 是 Kubernetes 能创建和管理的最小可部署计算单元，但实际工作里通常不直接手写裸 Pod，而是用 **Deployment** 管理 Pod。Deployment 会创建 ReplicaSet，ReplicaSet 再创建和维持 Pod。

![Image](<../.gitbook/assets/authcode (2)>)

### 3.1 用命令创建一个 Deployment

```bash
kubectl create deployment nginx --image=nginx:1.25
```

```bash
kubectl get deployments
kubectl get rs
kubectl get pods
```

你会看到 Pod 名字类似 `nginx-5869d7778c-k2x7p`，而不是单纯的 `nginx`。这说明它不是你直接创建的裸 Pod，而是 ReplicaSet 根据 Deployment 的 Pod 模板创建出来的。

### 3.2 看控制关系

```bash
kubectl describe pod <pod-name>
kubectl describe rs <replicaset-name>
```

在 Pod 详情里通常能看到 `Controlled By: ReplicaSet/...`；在 ReplicaSet 详情里能看到 `Controlled By: Deployment/nginx`。

### 3.3 扩容到 3 个副本

```bash
kubectl scale deployment nginx --replicas=3
kubectl get deployment nginx
kubectl get pods
```

这里你改的是 Deployment 的 `spec.replicas`。Deployment 把期望副本数交给 ReplicaSet，ReplicaSet 发现当前 Pod 不够，就创建新的 Pod。

### 3.4 验证 Pod 自愈

```bash
kubectl delete pod <one-nginx-pod-name>
kubectl get pods -w
```

准确地说，不是旧 Pod 复活了，而是 ReplicaSet 创建了一个新的 Pod 来补齐副本数。Pod 是可替换的，不要把单个 Pod 当成长期稳定实体。

### 3.5 滚动更新和回滚

```bash
# 查看当前镜像
kubectl get deployment nginx -o jsonpath='{.spec.template.spec.containers[0].image}{"\n"}'

# 更新镜像。示例故意使用明确版本，不使用 latest
kubectl set image deployment/nginx nginx=nginx:1.27

# 查看发布状态
kubectl rollout status deployment/nginx

# 查看发布历史
kubectl rollout history deployment/nginx
```

滚动更新时，Deployment 会创建新的 ReplicaSet，逐步增加新版本 Pod，同时逐步减少旧版本 Pod。

```bash
kubectl rollout undo deployment/nginx
kubectl rollout status deployment/nginx
```

### 3.6 更接近实际工作的 YAML 写法

命令式操作适合理解概念，但真实项目更常见的是把资源写成 YAML，然后用 `kubectl apply` 管理。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
  labels:
    app: nginx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
        - name: nginx
          image: nginx:1.27
          ports:
            - containerPort: 80
```

```bash
kubectl apply -f nginx-deployment.yaml
kubectl get deployment nginx
kubectl get rs
kubectl get pods
```

`selector.matchLabels.app=nginx` 和 Pod 模板里的 `labels.app=nginx` 必须匹配。这里 `app` 是 label key，`nginx` 是 label value。

***

## 4 访问应用：Service 和 Ingress

Pod 会被删除、重建、迁移，IP 也会变化。Service 的作用是给一组 Pod 提供稳定访问入口。它不创建 Pod，而是通过 label selector 找到后端 Pod。

### 4.1 先看 Pod 标签

```bash
kubectl get pods --show-labels
```

你可能会看到类似 `app=nginx,pod-template-hash=xxxxx`。在 `app=nginx` 里，`app` 是 label key，`nginx` 是 label value。Service selector 会用这个标签选择后端 Pod。

### 4.2 用 kubectl expose 创建 Service

```bash
kubectl expose deployment nginx --port=80 --target-port=80 --type=NodePort
kubectl get svc nginx
```

```
NAME    TYPE       CLUSTER-IP      EXTERNAL-IP   PORT(S)        AGE
nginx   NodePort   10.96.123.45    <none>        80:31234/TCP   20s
```

现在有两个常见访问方式：集群内部可以访问 `nginx:80` 或 `10.96.123.45:80`；集群外部可以通过 `<NodeIP>:31234` 访问。

### 4.3 Service 类型对比

| 类型               | 适用场景               | 说明                                        |
| ---------------- | ------------------ | ----------------------------------------- |
| **ClusterIP**    | 集群内部访问             | 默认类型，只在集群内暴露                              |
| **NodePort**     | 本地练习或临时调试          | 在每个 Node 上开一个端口，默认范围通常是 30000-32767       |
| **LoadBalancer** | 云上对外暴露服务           | 通常由云厂商创建外部负载均衡器                           |
| **Ingress**      | HTTP/HTTPS 域名和路径路由 | 需要额外的 Ingress Controller，例如 nginx-ingress |

### 4.4 Service YAML 写法

```yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx
spec:
  type: NodePort
  selector:
    app: nginx
  ports:
    - name: http
      port: 80
      targetPort: 80
```

```bash
kubectl apply -f nginx-service.yaml
kubectl get svc nginx
kubectl get endpoints nginx
```

如果 Service 能创建但访问不到，优先检查 selector 是否匹配 Pod label。selector 写错时，Service 没有后端 endpoints。

***

## 5 配置、密钥、健康检查和资源限制

能启动应用只是第一步。真实服务还需要把配置和镜像解耦，声明健康检查，并限制资源使用，避免一个容器拖垮节点。

### 5.1 ConfigMap 和 Secret

**ConfigMap**

存放非敏感配置，例如环境名、开关、普通配置文件。

**Secret**

存放密码、Token、证书等敏感信息。默认表现是 base64 编码，不等于加密。

```bash
kubectl create configmap app-config \
  --from-literal=APP_ENV=local \
  --from-literal=LOG_LEVEL=info

kubectl create secret generic app-secret \
  --from-literal=DB_PASSWORD='change-me'
```

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-with-config
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nginx-with-config
  template:
    metadata:
      labels:
        app: nginx-with-config
    spec:
      containers:
        - name: nginx
          image: nginx:1.27
          env:
            - name: APP_ENV
              valueFrom:
                configMapKeyRef:
                  name: app-config
                  key: APP_ENV
            - name: DB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: app-secret
                  key: DB_PASSWORD
```

不要把真实密码、Token、证书直接提交到 Git。生产环境通常会结合云厂商 KMS、External Secrets、Sealed Secrets 或 GitOps 密钥方案。

### 5.2 健康检查：readinessProbe 和 livenessProbe

| 探针                 | 作用             | 失败后会怎样                     |
| ------------------ | -------------- | -------------------------- |
| **readinessProbe** | 判断 Pod 是否可以接流量 | 失败时从 Service endpoints 中摘除 |
| **livenessProbe**  | 判断容器是否还活着      | 失败超过阈值后重启容器                |
| **startupProbe**   | 给慢启动应用更长启动时间   | 启动成功前暂缓其他探针判断              |

```yaml
readinessProbe:
  httpGet:
    path: /
    port: 80
  initialDelaySeconds: 3
  periodSeconds: 5
livenessProbe:
  httpGet:
    path: /
    port: 80
  initialDelaySeconds: 10
  periodSeconds: 10
```

### 5.3 资源请求和限制

`requests` 用于调度：告诉调度器这个容器至少需要多少 CPU 和内存。`limits` 用于限制：告诉运行时这个容器最多能用多少资源。

```yaml
resources:
  requests:
    cpu: "100m"
    memory: "128Mi"
  limits:
    cpu: "500m"
    memory: "256Mi"
```

后面做 HPA 时，CPU 利用率通常基于 `requests.cpu` 计算。没有 requests 的工作负载，很容易出现 HPA 无法判断或指标不稳定的问题。

***

## 6 自动扩缩容：metrics-server 和 HPA

HPA，全称 Horizontal Pod Autoscaler，用来根据 CPU、内存或自定义指标自动调整副本数。它依赖指标来源，本地环境最常见的是 metrics-server。

### 6.1 确认 metrics-server 是否可用

```bash
kubectl top nodes
kubectl top pods
```

如果报错 `metrics not available`，可以安装 metrics-server：

```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# 本地集群可能需要跳过 kubelet TLS 校验
kubectl -n kube-system patch deploy metrics-server --type=json \
  -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'

# 等 30 到 60 秒后重试
kubectl top nodes
```

### 6.2 创建 HPA

```bash
kubectl autoscale deployment nginx \
  --cpu-percent=70 \
  --min=1 \
  --max=5

kubectl get hpa
```

这表示：当 nginx Deployment 的平均 CPU 使用率超过目标值 70% 时，HPA 可以把副本数从 1 扩到最多 5。

本地 nginx 静态服务不一定能稳定触发 CPU 扩容。HPA 练习的重点是理解前提：metrics-server 可用、Deployment 设置了 CPU requests、HPA 能读取指标并改写 replicas。

***

## 7 工具链：kubectl、Helm、Terraform 怎么分工

| 工具            | 定位                             | 常见命令                                                   |
| ------------- | ------------------------------ | ------------------------------------------------------ |
| **kubectl**   | 直接操作 Kubernetes 资源             | `kubectl apply`、`kubectl get`、`kubectl describe`       |
| **Helm**      | 把一组 YAML 打包成可安装、可升级、可回滚的 Chart | `helm install`、`helm upgrade`、`helm rollback`          |
| **Terraform** | 管理基础设施，例如 VPC、云集群、节点组、负载均衡器    | `terraform plan`、`terraform apply`、`terraform destroy` |

### 7.1 Helm 最小理解

当应用越来越复杂时，手写和复制多份 YAML 会很难维护。Helm 的 Chart 可以把 Deployment、Service、Ingress、ConfigMap 等资源组织成一个包，再用 `values.yaml` 管理差异。

```bash
helm create my-chart
helm install my-app ./my-chart
helm upgrade my-app ./my-chart --values values.yaml
helm rollback my-app 1
helm uninstall my-app
```

```yaml
replicaCount: 3
image:
  repository: nginx
  tag: "1.27"
service:
  type: ClusterIP
  port: 80
```

### 7.2 Terraform 放在哪里

Terraform 不是 Kubernetes 自带组件，也不是 kubectl 插件。它更适合管理 Kubernetes 之外或更底层的基础设施，例如云服务器、VPC、EKS/ACK/GKE 集群、节点组、IAM、负载均衡器等。

![Image](<../.gitbook/assets/authcode (3)>)

简单分工：用 Terraform 建“集群和周边基础设施”，用 Helm 或 kubectl 部署“集群里的应用”。不要在入门阶段把两条线混在一起学。

***

## 8 清理练习环境

本文所有练习都放在 `lab` 命名空间里，最简单的清理方式是删除整个命名空间：

```bash
kubectl delete namespace lab
```

如果你不想删除命名空间，也可以逐个删除资源：

```bash
kubectl delete hpa nginx --ignore-not-found
kubectl delete svc nginx --ignore-not-found
kubectl delete deployment nginx nginx-with-config --ignore-not-found
kubectl delete configmap app-config --ignore-not-found
kubectl delete secret app-secret --ignore-not-found
```

***

## 9 参考

1. [Kubernetes Concepts](https://kubernetes.io/docs/concepts/)
2. [Kubernetes Hello Minikube](https://kubernetes.io/docs/tutorials/hello-minikube/)
3. [kubectl Quick Reference](https://kubernetes.io/docs/reference/kubectl/quick-reference/)
4. [Kubernetes Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
5. [Kubernetes Service](https://kubernetes.io/docs/concepts/services-networking/service/)
6. [Kubernetes ConfigMap](https://kubernetes.io/docs/concepts/configuration/configmap/)
7. [Kubernetes Secret](https://kubernetes.io/docs/concepts/configuration/secret/)
8. [Kubernetes Probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)
9. [Horizontal Pod Autoscaling](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
10. [Helm Quickstart](https://helm.sh/docs/intro/quickstart/)
11. [Terraform Introduction](https://developer.hashicorp.com/terraform/intro)
12. [Kubernetes 入门课程参考](https://guangzhengli.com/courses/kubernetes/pre)
