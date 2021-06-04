# <span style="color:blue">PROJETO: Análise de Sentimentos - Pull Requests</span>

<b>O problema</b>: Um problema comum em NLP é a classificação de amostras de texto, isto é, dado um texto, alocar em uma categoria, ou medir um índice de afinidade para uma ou mais categorias. Um problema comum em grandes projetos de código aberto é determinar quando o comentário em um pull request é positivo ou negativo.

<b>O objetivo</b>: Construir um modelo para classificação de comentários de pull-requests, indicando se podem ser consideradas <b>positivas</b>, <b>negativas</b> ou <b>neutras</b>.

### Organização do projeto
Existem duas pastas com os sub-projetos deste desafio:

* trainer: Possui o código para o treinamento do modelo
* API: Possui o código para expor o modelo como API

Em cada um desses sub-projetos existe um arquivo README.md com instruções específicas.

### Premissas do projeto
Para este desafio fora estabelecidas algumas premissas:

* Foi utilizado um modelo pré-treinado para análise de sentimentos (https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment). Esse modelo foi treinado em um grande conjunto de dados de Twitter para análise de sentimentos, e o conhecimento adquirido pode ser transferido para o problema proposto;
* A decisão de utilizar um modelo pré-treinado é porque os modelos de linguagem atualmente são considerados estado da arte para esse tipo de tarefa (http://nlpprogress.com/english/sentiment_analysis.html). Contudo, isso tem um maior custo computacional, pois são modelos maiores e precisam de maior poder computacional para o treinamento e também para inferência. Outras opções poderiam ter sido avaliadas, contudo essa opção para o teste pareceu viável pois é um modelo que uso na minha pesquisa de mestrado.
* O problema proposto são para dados de pull requests, contudo o modelo base utilizado foram de dados do Twitter, ou seja, será realizada uma tranferência de aprendizado de domínios diferentes (cross-domain). Em um cenário real, poderia ser avalidado de se fazer anotações no próprio domínio do problema, contudo para um teste é esperado que essa abordagem funcione bem.
* Foi utilizado um dataset público (https://www.kaggle.com/crowdflower/twitter-airline-sentiment) com três classes, para o treinamento do modelo supervisionado, pois o modelo indicado só possuia duas classes. Uma vez que o modelo foi treinado para ter duas saídas não é possível utilizá-lo para três, conforme solicitado no problema, por isso foi utilizado esse outro dataset.
* O dataset de comentários não foi utilizado, pois no meu entendimento pela descrição do problema uma possibilidade é que esse dataset fosse utilizado para o treinamento de vetores de representações (Embeddings), contudo se o intuíto de fato for esse, um treinamento de abordagens como Word2Vec poderia não ser tão efetiva por ser vetores estáticos, ou seja, a representações são sempre fixas independente do contexto. Por isso, a decisão de usar um modelo de linguage é que as representações vetoriais são contextuais. É importante deixar claro que em um cenário real, as duas opções e outras poderiam ser avaliadas para decidir qual abordagem seria mais adequada para resolver o problema, contudo para conclusão do teste foi escolhida essa abordagem por inicialmente parecer mais adequada.