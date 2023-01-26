---
title: "Git配置多个ssh用户"
date: 2016-08-04T01:40:34+08:00
draft: false
tags:
  - git
ShowToc: true
---

今天这是我写的第一篇博客，不知道在使用 git 中遇到过下面的问题 :

自己配置了全局的用户信息，然后我们却有两个或两个之上 的 git 账号在不同的服务器上(比如
github, csdn),或者我们正在做两个以上项目(在 github)需要配置不同 ssh， 我们的用户信息却
不同, 但是我们却不能配置两个全局信息。为了使用 ssh 服务，避免每次都要输入用户密码，我们
还是要使用秘钥来配置。

## 其实这就是个坑

当我们是新手时，每次看到教程都要配置全局的个人信息(user.name, user.email)，虽然它比较简单，但确实导致
了一些使用的问题。 切入正题，解决方案：

1. 我们不要使用使用全局的用户信息配置，改成在项目内部配置。

    ```shell
    git config user.name "username"
    git config user.email "XXX@email.com"
    ```

2. 配置公钥时，需要在`~/.ssh`目录下添加一个 `config` 的文件,格式如下

   ```shell
    Host 名称(自己决定，方便输入记忆的)
    HostName 主机名
    User 登录的用户名
    IdentityFile 私钥地址
   ```

3. 例子

    ```shell
    #github "注释
    Host github.com " 简称
    HostName github.com " 服务器地址
    User chalme " 用户名
    IdentityFile ~/.ssh/id_rsa " 公钥地址

    Host code.csdn.net
    HostName code.csdn.net
    User chalme
    IdentityFile ~/.ssh/id_rsa_csdn

    Host A
    HostName github.com
    User B
    IdentityFile ~/.ssh/id_rsa_csdn
    ```
