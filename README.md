# On-premise AIOps Practice
This is the guide for deploying AIOps within Kubernetes.


## Getting Started
You need a Kubernetes cluster.
**Note:** As a pre-condition, get open source container images and upload to your local repository.


### Test It Out
1. Create monitor namesapce:
```sh
$ kubectl create ns monitor
```

2. Install monitor tools under the created ns(`monitor`):
```sh
$ kubectl apply -f Inside_SUT_deployment.yml -n monitor
```

3. Create aiops namesapce:
```sh
$ kubectl create ns aiops
```

4. Deploy centilized monitor storage and dashboard:
```sh
$ kubectl apply -f AIOps_deploy.yml -n aiops
```

5. Deploy pytorch environment for log anomaly training and predict:
```sh
$ kubectl apply -f torch.yml -n aiops
```
**NOTE:** package your tools to container image, replace it to related deployment yml



### Undeploy UnDeploy on-premise AIOps:
```sh
$ bash delete_aiops.sh
```

