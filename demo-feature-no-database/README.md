# Configuração do Projeto

## Criação da Pasta Git

### Iniciar a Criação da Pasta Git
1. No canto superior direito da visualização do Workspace, clique na seta para baixo ou no menu Adicionar (Create)
2. Selecione a opção Pasta Git (Git folder) no menu suspenso

### Preencher os Dados do Repositório
Na janela de criação, complete os seguintes campos:
- **URL do repositório Git**: Cole `https://github.com/Databricks-BR/workshop_agents.git`
- **Provedor Git**: Selecione "GitHub" (ou o provedor apropriado)
- **Nome da pasta**: Digite `demo-llm` (esse será o nome exibido dentro do seu workspace)
- **(Opcional) Padrão de checkout enxuto**: Deixe em branco salvo se quiser baixar apenas parte do repositório

### Autenticação com o Provedor Git
Para repositórios públicos, geralmente não é necessário autenticação. Para privados, certifique-se de configurar suas credenciais na aba de configurações do Databricks. Siga os prompts caso seja necessário realizar login ou fornecer um token de acesso.

### Criar a Pasta Git
1. Clique em **Criar pasta Git**
2. O Workspace criará uma pasta chamada `demo-llm` com o conteúdo do branch main do repositório informado

### Utilizar a Pasta Git
Agora você pode abrir, editar, adicionar notebooks e arquivos, e realizar operações Git (commit, pull, push, gerenciamento de branches, etc.) diretamente pela interface do Workspace. Para ações Git, utilize o diálogo Git disponível nos notebooks ou clicando com o botão direito no nome da pasta, selecionando **Git...**.

## Configuração do Arquivo .env

### Criar o Arquivo .env
1. Clone o arquivo `config_simple.example.env`
2. Clique com o botão direito e selecione **clone**
3. Mude o nome de "(Clone) config_simple.example.env" para apenas "`.env`"

### Comentar as Variáveis de Host e Token
1. Comente a linha da variável `DATABRICKS_HOST`
2. Comente a linha da variável `DATABRICKS_TOKEN`

### Configurar a Tool do Genie Space
1. Descomente a linha `UC_TOOL_NAMES`
2. Coloque `UC_TOOL_NAMES="main.default.chat_with_sales"`

## Criação do Catálogo "main"

### Acessar o Databricks Workspace
Entre no seu workspace do Databricks com as permissões adequadas (normalmente, você precisa ser administrador do metastore ou ter privilégio para criar catálogos).

### Abrir o Catalog Explorer
No menu lateral do Databricks, clique em **Catálogo** para abrir o Catalog Explorer.

### Iniciar a Criação de um Novo Catálogo
1. No Catalog Explorer, localize e clique no botão **+ Novo** ou **Adicionar**
2. Selecione a opção **Criar Catálogo**

### Preencher as Informações do Catálogo
1. No formulário de criação, insira `main` como o nome do catálogo
2. (Opcional) Adicione uma descrição para ajudar na identificação do objetivo do catálogo
3. (Dependendo do ambiente e configurações) Defina outras opções solicitadas, como localização de armazenamento default ou políticas de acesso

### Confirmar a Criação
1. Clique em **Criar** ou **Salvar**
2. O catálogo `main` será criado e aparecerá na lista de catálogos no Catalog Explorer

## Cadastro da Tool para o Genie Space

### Abrir o Notebook Genie Tools
1. Na pasta `others`, abra o notebook `genie_tools`
2. Conecte ao compute serverless
3. Clique em **connect** e selecione **Serverless**
4. Execute a primeira célula

## Criação de um Token de Acesso

### Acessar as Configurações do Usuário
1. Acesse seu workspace do Databricks Free Edition com suas credenciais de usuário
2. No canto superior direito, clique no ícone do seu usuário (avatar de perfil)
3. No menu suspenso, selecione **Configurações**

### Navegar até Tokens de Acesso
1. No painel de navegação à esquerda, clique em **Desenvolvedor**
2. Navegue até a seção **Tokens de Acesso** e clique em **"Manage"**

### Gerar Novo Token
1. Clique no botão **Gerar Novo Token**
2. Opcionalmente, forneça uma descrição ou nome para o token
3. Defina uma data de expiração para o token se necessário
4. Clique em **Gerar** ou **Criar**

### Salvar o Token
Assim que o token for gerado, ele será exibido apenas uma vez. **Copie e salve o token em um local seguro imediatamente** - não será possível recuperá-lo depois!

## Obtenção do Genie ID

### Acessar o Espaço Genie
1. Abra a interface do Genie no seu workspace Databricks
2. Navegue até o espaço Genie desejado (e.g. Bakehouse Sales Starter Space)
3. Selecione-o na barra lateral do Genie ou utilize qualquer botão com o rótulo "Abrir Espaço Genie"

### Extrair o ID da URL
1. Verifique a barra de endereços do navegador
2. O formato da URL será: `https://<databricks-instance>/genie/rooms/<space-id>`
3. A cadeia de caracteres após `/genie/rooms/` é o ID do seu espaço Genie

**Exemplo:**
Se a sua URL for:
`https://example.databricks.com/genie/rooms/12ab345cd6789000ef6a2fb844ba2d31`

Então `12ab345cd6789000ef6a2fb844ba2d31` é o ID do espaço Genie.

**Alternativa:** O ID do espaço Genie também pode ser encontrado na aba de configurações ("Settings") do espaço Genie.

## Identificação do Host Databricks

Para identificar o host Databricks, observe a URL completa na barra de endereços do navegador quando você está acessando o workspace Databricks. O host é a parte principal do endereço, geralmente logo após o "https://".

**Exemplos:**
- Se a URL for `https://meu-workspace.cloud.databricks.com`, o host é `https://meu-workspace.cloud.databricks.com`
- Se a URL for `https://adb-1234567890123456.7.azuredatabricks.net`, o host é `https://adb-1234567890123456.7.azuredatabricks.net`

Esse endereço é o que você deve usar como "host" ao configurar integrações, APIs ou conexões externas.

## Configuração Final da Genie como Tool

### Configurar o Notebook
1. Abra a pasta `others`
2. Abra o notebook chamado `genie_tools` (e.g. `/Workspace/Users/user@gmail.com/demo/others/genie_tools`)
3. Adicione o host no espaço indicado na última célula
4. Adicione o token obtido anteriormente no espaço indicado na última célula
5. Adicione o ID do Genie space no espaço indicado

### Executar a Função
Execute a segunda célula com a function `chat_with_sales` depois que os identificadores forem inseridos.

## Criação de um Databricks App

### Acessar a Área de Apps
1. Faça login no seu workspace Databricks
2. No menu lateral, clique em **Compute**
3. Procure e selecione a seção **Apps**
4. Clique em **Criar App** ("Create App")

### Configurar o App
1. **Escolha o Tipo**: Criar do zero para ter controle total sobre a estrutura de arquivos e lógica do app
2. **Nome**: Defina um nome para seu app, por exemplo `demo-llm`
3. Clique em **Criar** ("Create app")
4. Aguarde o provisionamento do compute serverless automático
5. Espere o botão **Deploy** ficar disponível

## Configuração do Service Principal

### Localizar o Service Principal
1. Na página de detalhes do app, procure pela aba **"Autorização"** ou **"Permissões"**
2. Dentro da aba de Autorização, localize as informações sobre "Autorização do App"
3. Copie o **client ID OAuth do service principal do app** (também chamado de application ID ou client ID)

## Grant para o Principal do Databricks App

### Configurar Permissões no Catálogo
1. Abra o **Catalog Explorer** clicando no ícone Catálogo na barra lateral
2. Localize o catálogo `main` e clique no nome para abrir os detalhes
3. Selecione a aba **Permissões**
4. Clique no botão **Conceder** ("Grant")

### Adicionar Permissões
1. No campo de principal, insira o **Application ID** (client ID) do service principal copiado anteriormente
2. Selecione as permissões desejadas:
   - Para acesso total: escolha **TODOS OS PRIVILÉGIOS** ("ALL PRIVILEGES")
   - Ou selecione permissões específicas como SELECT, MODIFY, EXECUTE
3. Confirme clicando em **OK**

## Deploy Final do App

### Realizar o Deploy
1. Abra novamente as configurações do Databricks App: **Compute** → **Apps** → "nome do App" (e.g. `demo-llm`)
2. Clique em **Deploy**
3. Selecione a pasta `demo-llm` (ou a pasta com todos os arquivos do App) e selecione **"Deploy"**
4. Aguarde o deploy ser concluído
5. Clique no link ao lado de **Running**

---

*Projeto configurado com sucesso! O aplicativo Databricks está agora em execução com integração Git e ferramentas Genie configuradas.*