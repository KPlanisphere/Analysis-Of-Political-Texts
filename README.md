# SISTEMA DE AN¨¢LISIS POL¨ªTICO
## RECUPERACI¨®N DE LA INFORMACI¨®N: PROYECTO FINAL

 Las t¨¦cnicas automatizadas de an¨¢lisis de textos han asumido un papel cada vez m¨¢s importante en el estudio de los partidos y el discurso pol¨ªtico. Los investigadores han estudiado manifiestos, discursos en el parlamento y debates en reuniones nacionales de partidos. Estos m¨¦todos han demostrado ser muy prometedores para medir las caracter¨ªsticas latentes de los textos. Sin embargo, en su aplicaci¨®n, los modelos de escala requieren una gran cantidad de decisiones por parte del investigador que probablemente tengan implicaciones sustanciales para el an¨¢lisis. Mar¨ªa Ramos se dedica al an¨¢lisis de textos pol¨ªticos y los ha contratado para realizar un an¨¢lisis de textos pol¨ªticos. Piensa que un punto de partida podr¨ªan ser los informes presidenciales desde Carlos de Salinas hasta la fecha. Los informes pueden ser localizados en la red. Sin embargo, el sistema no debe restringirse a estos y debe poder utilizarse con cualquier texto pol¨ªtico.
 
Est¨¢ interesada en realizar gr¨¢ficas y conocer por ejemplo que temas abordaron m¨¢s cada uno de los presidentes (autores) bajo estudio. Si hablaron de mejorar el salario, de ganancia petroleras, de salud, etc.

Tambi¨¦n est¨¢ interesada en identificar semejanzas y diferencias entre los representantes de diferentes partidos. 
Mar¨ªa ha pensado en las siguientes funciones para su sistema:

### Estad¨ªsticas Generales
Aqu¨ª se presentar¨ªan datos generales de los documentos tales como longitud de cada documento (n¨²mero de palabras). N¨²mero de palabras diferentes de los textos. N¨²mero de documentos analizados por pol¨ªtico y otras que se juzgue de utilidad.

### Gr¨¢fica de dispersi¨®n l¨¦xica

La importancia de una palabra/token puede estimarse por su dispersi¨®n en el corpus. Los tokens var¨ªan en su distribuci¨®n a lo largo del texto, lo que indica cu¨¢ndo o d¨®nde se utilizan ciertos tokens en el texto. La dispersi¨®n l¨¦xica es una medida de la homogeneidad de la palabra en todo el corpus. Se pueden generar distribuciones de palabras para tener una idea general de los temas, su distribuci¨®n y sus cambios. Un gr¨¢fico de dispersi¨®n l¨¦xica representa las apariciones de la palabra y la frecuencia con la que aparecen desde el comienzo del corpus. Por tanto, los diagramas de dispersi¨®n l¨¦xica son ¨²tiles para identificar patrones.

El eje x del gr¨¢fico de dispersi¨®n l¨¦xica muestra el desplazamiento de palabras, es decir, la aparici¨®n y frecuencia de las palabras a lo largo de los discursos y el eje y muestra los problemas espec¨ªficos.

### Modelos de serie tiempo
En el an¨¢lisis predictivo, el tiempo es un factor muy importante que debe tenerse en cuenta. El patr¨®n reconocido o predicho debe estudiarse y verificarse con respecto al tiempo. Los datos de series de tiempo no son m¨¢s que una serie de datos ordenados en el tiempo. La figura 5 captura la tendencia comparativa de las series temporales entre los temas ¡°paz¡± y ¡°terrorismo¡± como ejemplo.

### Representaci¨®n en WordCloud de los discursos
Para obtener una impresi¨®n r¨¢pida y hol¨ªstica de las transcripciones de discursos que se est¨¢n considerando, se crean nubes de palabras. Las nubes de palabras son una t¨¦cnica de visualizaci¨®n sencilla e intuitiva que se utiliza a menudo para proporcionar una primera impresi¨®n de documentos de texto. WordClouds muestra las palabras m¨¢s frecuentes del texto como una lista ponderada de palabras en un dise?o espacial espec¨ªfico, por ejemplo, dise?o secuencial, circular o aleatorio. Los tama?os de fuente de las palabras cambian seg¨²n su relevancia y frecuencia de aparici¨®n, y otras propiedades visuales como el color, la posici¨®n y la orientaci¨®n a menudo var¨ªan por razones est¨¦ticas o para codificar visualmente informaci¨®n adicional.

### Examinar tendencias/patrones utilizando diagramas de barras
Los gr¨¢ficos de barras representan c¨®mo se distribuyen los datos entre ciertos valores potenciales. Aunque es un gr¨¢fico de apariencia simple, un gr¨¢fico de barras tiene la capacidad de capturar la esencia de los datos al juzgar la dispersi¨®n y responder ciertas preguntas. La figura que se muestra a continuaci¨®n es una representaci¨®n del "Nombre del tema frente al n¨²mero de menciones". Para cada nombre de token proporcionado, se calcula la frecuencia de aparici¨®n y se genera el gr¨¢fico.

### Clasificador
Mar¨ªa desea caracterizar escritos de la pol¨ªtica mexicana como Neoliberales y perteneciente al Humanismo M¨¦xico. As¨ª que tambi¨¦n desea que el sistema sea capaza de diferenciarlos. Mar¨ªa ha solicita una propuesta de lo que su sistema podr¨ªa proporcionarle de acuerdo con las funciones que describe. Tambi¨¦n espera, de ser posible, que le proporciones funciones adicionales que puedan serle de utilidad, por ejemplo, la detecci¨®n de conceptos. Est¨¢ abierta a cualquier propuesta.


