
## Datos vs Modelo
- Para poder analizar un modelo, o conjunto de datos no basta con esperar que todo cuadre.
- Debemos conocer los valores que acepta y devuelve el modelo, tanto en inputs como variables para la evaluación.
- En este punto debemos cuestionarnos, el dataset que tenemos es para evaludar un modelo o para evaluar al dataset? suena raro, pero es importante.
- Según la naturaleza del objetivo podremos adaptar el dataset al modelo o el modelo al dataset.

## Futuro de Tokenización

- Modelos byte-level (ByT5, CANINE) resuelven multilingüismo y OOV
- Limitación principal: 3-5x más lentos sin mejoras claras en accuracy
- Para este proyecto, subword (WordPiece) sigue siendo la mejor opción
- Byte-level relevante para: idiomas raros, texto ruidoso, code-switching

## Validación de Explicaciones

- Fidelidad: mide qué tan bien una explicación refleja el comportamiento del modelo
- Test de eliminación: eliminar la palabra más importante según SHAP debería reducir la confianza en la predicción. Pero no esta simple. Se debe definir "eliminar" y "palabra" correctamente.
- Acuerdo entre métodos: usar correlación de Spearman para comparar rankings de importancia entre SHAP y LIME
- Al buscar experimentar con los cambios de palabras fue donde mas variables a considerar encontre. Llegado al punto de porque un #. tiene mas valor en SHAP que una Terrible. Y ocurre que segun el tokenizador, podrias obtener tokens que por si solo no tienen sentido, pero en conjunto si, ocacionando que un #. le gane a #Ter o #rible. 
- Este proble del tokenizador deriba en que uno debe definir en que nivel se va a trabajar, si en tokens, palabras, oraciones, etc. y como se va a eliminar una palabra, si reemplazandola por un token especial o por una palabra aleatorea.
- Resumen: 
    - Define tokenización y eliminación claramente
    - Nivel de análisis: tokens, palabras, oraciones
    - Usa métricas cuantitativas y cualitativas

## Dashboard Interactivo
- No tengo idea de como luce uno similar, pero lo arme lo mas simple posible y amigable para pruebas "rapidas".
- Como trabajo con texto, ingresar texto manualmente es clave.
- Permitir seleccionar modelos abre la puerta a comprender mucho mejor las explicaciones.
- Comparar SHAP y LIME visualmente ayuda a entender diferencias y similitudes.
- Validación de explicaciones integrada permite evaluar calidad de explicaciones en tiempo real.

## SHAP vs LIME
- SHAP: basado en teoría de juegos, asigna importancia a características considerando todas las combinaciones posibles. Consistente y con fundamentos teóricos sólidos. Requiere recursos computacionales.
- Devuelve valores por cada token, lo que puede ser más detallado pero también más ruidoso. Permitiendo hacer mas experimentos con la importancia de cada token.
- LIME: crea un modelo local interpretable alrededor de la predicción. Más rápido y flexible, pero menos consistente. Puede ser inestable con pequeñas perturbaciones.
- Devuelve importancia a nivel de palabra, lo que puede ser más intuitivo pero menos detallado.

- Creo que lo ideal es usar ambos, ya que SHAP tiene una base teórica mas solida, pero LIME es mas rapido y flexible. Usando ambos podemos obtener una visión mas completa de las explicaciones del modelo. A grandes rasgos, y una vez que contamos con estas metricas, podemos hacer una validacion de explicaciones mas robusta.

Resumen         |       SHAP             |       LIME
Devuelve        | Array numpy por token  | Lista de (palabra, valor)
Necesitas procesar | No, solo .values[0] | extract_lime_values()
Tokens que devuelven | Todos los tokens       | Solo palabras