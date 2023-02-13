---
title: "DDD"
date: 2023-02-13T09:04:13+08:00
draft: false
tags:
  - think
  - DDD
ShowToc: true
---

[领域驱动设计( DDD )](https://en.wikipedia.org/wiki/Domain-driven_design) 是一种主要的软件设计方法，侧重于根据领域专家的输入对软件进行建模以匹配该领域。在领域驱动设计下，软件代码的结构和语言（类名、类方法、类变量）要与业务领域相匹配。

领域驱动设计基于以下目标：

- 将项目的主要重点放在核心领域和领域逻辑上；
- 基于域模型的复杂设计；
- 在技 ​​ 术专家和领域专家之间发起创造性协作，以迭代方式完善解决特定领域问题的概念模型。

## 特点

1. 角色复杂性。 解决软件系统复杂度的一种建模工具。
2. 业务本身复杂，新同学学习和理解成本高。
3. 支撑多业务，多业务的支撑需要平衡共性和差异性的难题，很容易造成相似代码到处拷贝。
4. 模型设计和实现能直接表现业务语义
5. 真正拥抱面向对象的思想

## 市场上的应用架构

### 传统 MVC 三层架构

### 洋葱架构

![PHuGHh](http://qiniu.chalme.top/blog/20230213/PHuGHh.jpg)
同心圆代表软件的不同部分，从里向外依次是领域模型，领域服务，应用服务和外层的基础设施和用户终端。

洋葱架构根据依赖原则，定义了各层的依赖关系，越往里依赖程度越低，代码级别越高，越是核心能力。外圆代码依赖只能指向内圆，内圆不需要知道外圆的情况，洋葱架构也是典型的分层架构，和 DDD 分层架构一样，都体现了**高内聚，低耦合**的设计特性。

### CQRS 架构

CQRS 是“命令查询责任分离”（Command Query Responsibility Segregation）的缩写。
在基于 CQRS 的系统中，命令(写操作)和查询(读操作)所使用的数据模型是有区别的。
![FH9aIA](http://qiniu.chalme.top/blog/20230213/FH9aIA.jpg)

**为什么将查询和命令区隔开呢？**
为什么将查询和命令区隔开呢，是因为在实现各种各样的查询操作时，往往要求非常灵活，多个领域对象的关联查询、分页查询，往往是每个对象取几个字段组成一个视图模型，与领域知识关系没有太紧密的关系。

## 应用架构

### 应用架构的意义

- 定义一套良好的结构；
- 治理应用复杂度，降低系统熵值；
- 从随心所欲的混乱状态，走向井井有条的有序状态。

应用架构在不同的地方有不同的理解，不同的业务可能也会有所区别。 能实现目的意义的结构就是一种好的架构。

应用架构有很多模式

- [COLA](https://github.com/alibaba/COLA)
- ...

### 一种应用架构模式

![iXGPEt](http://qiniu.chalme.top/blog/20230213/iXGPEt.jpg)

我们分了 API、Presentation、Application、Domain、Adapter、Infrastructure 6 大模块。上面箭头方向不是控制流，而是模块依赖方向。由于依赖可以传递，所以 Api 模块也就被传递到 Presentation 依赖，Domain 层也就被传递到 Infrastructure 依赖。

**API**
主要是对外提供的远程 RPC 接口，包含了接口、参数和返回值的定义。API 模块不依赖任何本项目的其他模块，只依赖基础平台提供的基础公共包 ihr-platform-common（主要是提供返回值 wrapper,领域层的基础领域接口），不依赖其他任何包，保持干净，不为依赖应用引入包版本冲突问题。
查询参数命名以为 Query 结尾，命令参数以 Command 结尾，返回结果以 DTO 结尾。方法返回值需使用 Result 封装。因为是对外提供的 API，所以要有完备的 javadoc 注释。
**Presentation（对外接口表现层）**
包含 http 接口（web），hsf 接口实现（provider），定时调度（job），消息订阅（mqconsumer）。是我们对外提供数据服务的起点。本层不定义参数和数据转换对象（DTO）。从洋葱架构图上可以看出，Presentation 作为最外层模块之一，不被应用其他模块依赖。其只依赖 Application 模块。同时其实现 Api 中的远程接口，对 Api 模块的依赖通过 Application 传递。
**Application（业务场景逻辑编排层）**
主要处理业务逻辑的编排，CQRS 在这一层进行首先体现，分 command 包（处理命令逻辑），query 包（处理查询逻辑）。
command 包下的 CommandService 只会调用 domain 层下的领域或者仓储服务处理对象的增删改逻辑。
query 包下的 QueryService 首先调用 querytunnel 接口（数据查询通道）获取视图数据，querytunnel 的实现有基础设施层完成。QueryService 也可对查询逻辑进行编排，比如其很可能是从 querytunnel 中获取的一部分数据和从 Adapter 中获取的另一部分进行组装成一个视图 DTO 返回。
本层的业务处理如果是为了响应 Api 中的远程 hsf 接口，则其参数（Command、Query），返回结果（DTO）直接采用 Api 中定义的。如果是为响应 web 接口，则其参数和结果需要在 Application 层定义。可以参照上图例子。
**Domain（领域层）**
主要领域模型和聚合仓储服务的定义，这一层基本是按照 DDD 的思想组织代码逻辑，当然大家可以根据业务域的领域知识复杂度决定是采用贫血还是充血模式，比如，如果是简单的工具类应用，完全不必采用充血模式。这一层只依赖 Adapter（防腐层）。记住一个聚合只能有一个 Repository。

**Adapter（防腐层）**
主要是对远程服务和中间件服务的防腐，这一层定义的接口可以不用带中间件的语意。比如缓存中间件使用的是 Redis，这里定义的缓存接口可以是 CacheClient,。同时返回的对象是值对象（Value Object），不用带任何类型的 O 结尾。远程服务也一样的逻辑。由于这一层提供的能力可以被领域层用到，所以只被领域层依赖。
**Infrastructure（基础设施层）**
这一层主要是提供对以上层接口的实现，比如仓储接口（Repository），查询数据通道接口（QueryTunnel），远程服务接口（XxAdapter），中间件接口（XxClient）等等。下面目前分两个包 dal（数据访问）adapter（防腐）。dal 包下是 Repository 和 QueryTunnel 的实现，adapter 包下是远程服务和中间件的实现。外部返回的数据模型通过 converter 转换成领域模型，或者 QueryTunnel 的 DTO 返回给被依赖层。
Infrastructure 层只依赖 Application 层，其和 Presentation 都属于最外层的模块，不被其他任何模块依赖。又因为 Application 依赖了其他模块，所以其他模块的依赖都可以传递到 Infrastructure 层。

## 参考

https://github.com/alibaba/COLA
