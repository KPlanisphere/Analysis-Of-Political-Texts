# SISTEMA DE ANÁLISIS POLÍTICO
## RECUPERACIÓN DE LA INFORMACIÓN: PROYECTO FINAL

 Las técnicas automatizadas de análisis de textos han asumido un papel cada vez más importante en el estudio de los partidos y el discurso político. Los investigadores han estudiado manifiestos, discursos en el parlamento y debates en reuniones nacionales de partidos. Estos métodos han demostrado ser muy prometedores para medir las características latentes de los textos. Sin embargo, en su aplicación, los modelos de escala requieren una gran cantidad de decisiones por parte del investigador que probablemente tengan implicaciones sustanciales para el análisis. María Ramos se dedica al análisis de textos políticos y los ha contratado para realizar un análisis de textos políticos. Piensa que un punto de partida podrían ser los informes presidenciales desde Carlos de Salinas hasta la fecha. Los informes pueden ser localizados en la red. Sin embargo, el sistema no debe restringirse a estos y debe poder utilizarse con cualquier texto político.
 
Está interesada en realizar gráficas y conocer por ejemplo que temas abordaron más cada uno de los presidentes (autores) bajo estudio. Si hablaron de mejorar el salario, de ganancia petroleras, de salud, etc.

También está interesada en identificar semejanzas y diferencias entre los representantes de diferentes partidos. 
María ha pensado en las siguientes funciones para su sistema:

### Estadísticas Generales
Aquí se presentarían datos generales de los documentos tales como longitud de cada documento (número de palabras). Número de palabras diferentes de los textos. Número de documentos analizados por político y otras que se juzgue de utilidad.

### Gráfica de dispersión léxica

La importancia de una palabra/token puede estimarse por su dispersión en el corpus. Los tokens varían en su distribución a lo largo del texto, lo que indica cuándo o dónde se utilizan ciertos tokens en el texto. La dispersión léxica es una medida de la homogeneidad de la palabra en todo el corpus. Se pueden generar distribuciones de palabras para tener una idea general de los temas, su distribución y sus cambios. Un gráfico de dispersión léxica representa las apariciones de la palabra y la frecuencia con la que aparecen desde el comienzo del corpus. Por tanto, los diagramas de dispersión léxica son útiles para identificar patrones.

El eje x del gráfico de dispersión léxica muestra el desplazamiento de palabras, es decir, la aparición y frecuencia de las palabras a lo largo de los discursos y el eje y muestra los problemas específicos.

### Modelos de serie tiempo
En el análisis predictivo, el tiempo es un factor muy importante que debe tenerse en cuenta. El patrón reconocido o predicho debe estudiarse y verificarse con respecto al tiempo. Los datos de series de tiempo no son más que una serie de datos ordenados en el tiempo. La figura 5 captura la tendencia comparativa de las series temporales entre los temas “paz” y “terrorismo” como ejemplo.

### Representación en WordCloud de los discursos
Para obtener una impresión rápida y holística de las transcripciones de discursos que se están considerando, se crean nubes de palabras. Las nubes de palabras son una técnica de visualización sencilla e intuitiva que se utiliza a menudo para proporcionar una primera impresión de documentos de texto. WordClouds muestra las palabras más frecuentes del texto como una lista ponderada de palabras en un dise?o espacial específico, por ejemplo, dise?o secuencial, circular o aleatorio. Los tama?os de fuente de las palabras cambian según su relevancia y frecuencia de aparición, y otras propiedades visuales como el color, la posición y la orientación a menudo varían por razones estéticas o para codificar visualmente información adicional.

### Examinar tendencias/patrones utilizando diagramas de barras
Los gráficos de barras representan cómo se distribuyen los datos entre ciertos valores potenciales. Aunque es un gráfico de apariencia simple, un gráfico de barras tiene la capacidad de capturar la esencia de los datos al juzgar la dispersión y responder ciertas preguntas. La figura que se muestra a continuación es una representación del "Nombre del tema frente al número de menciones". Para cada nombre de token proporcionado, se calcula la frecuencia de aparición y se genera el gráfico.

### Clasificador
María desea caracterizar escritos de la política mexicana como Neoliberales y perteneciente al Humanismo México. Así que también desea que el sistema sea capaza de diferenciarlos. María ha solicita una propuesta de lo que su sistema podría proporcionarle de acuerdo con las funciones que describe. También espera, de ser posible, que le proporciones funciones adicionales que puedan serle de utilidad, por ejemplo, la detección de conceptos. Está abierta a cualquier propuesta.


