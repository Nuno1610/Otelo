Proyecto IAIS 25-26: Búsqueda adversaria - Aprendiendo a jugar a Otelo

Nuno José del Pino Escalante y Javier Gutiérrez Pastor

La entrega se realiza en un solo archivo comprimido ya que la plataforma no permite adjuntar más de un archivo, resultando imposible separar documentación y código.

El proyecto se estructura en 2 bloques principales:
- docs: Contiene la documentación del proyecto. Está disponible tanto en formato Word(.docx) como en LATEX(.tex)
- src: Contiene todo el código fuente

Instrucciones:

Desde la carpeta src, se podrán ejecutar los archivos de la siguiente manera:

    Jugar al juego: python -m game.play_match
    Generar partidas para obtener datos: python -m game.game_generator
    Leer datos generados: python -m data.results_reader
    Generar modelo a partir de datos de entrenamiento: python -m agent.model.model

Nota: Si no se hace desde la carpeta src es muy probable que los scripts no se ejecutan correctamente.