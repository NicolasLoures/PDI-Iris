
# Cv2 principal interface do OpenCV em Python para processamento de imagens
# Numpy biblioteca para computação científica com Python, como tratamos imagens como arrays e matrizes, o numpy é essencial para manipulação de imagens
import cv2
import numpy as np

def filtro_mediana(img: np.ndarray, ksize: int=3) -> np.ndarray:
    """
    Aplica um filtro de mediana na imagem.

    Filtro Mediano: 
        Para cada pixel da imagem, o filtro de mediana considera sua vizinhança(janela de ksize x ksize) 
        Ordena os valores de intensidade dentro dessa janela e subsitui o pixel central pelo valor mediano.
        Isso ajuda a remover ruídos, especialmente ruídos sal e pimenta, preservando as bordas.

    Parâmetros:
    img (np.ndarray): Imagem de entrada. -> Espera um array numpy representando a imagem;
                                            Que pode ser colorida (3 canais) ou em escala de cinza (1 canal).
    ksize (int): Tamanho do kernel-> "janela" quadrada do filtro. Deve ser um número ímpar

    Retorna:
    np.ndarray: Outro array numpy representando a imagem filtrada.
    """
    # Verifica se o tamanho do kernel é ímpar
    if ksize % 2 == 0:
        raise ValueError("O tamanho do kernel deve ser um número ímpar.")
    
    # Aplica o filtro de mediana

    '''
    cv2.medianBlur() -> Recebe a imagem e o tamanho do kernel e retorna a imagem filtrada, lidando internaento com os contornos
    '''
    return cv2.medianBlur(img, ksize)



def equalizacao_histograma(img: np.ndarray) -> np.ndarray:
    """
    Aplica a equalização de histograma na imagem.

    Equalização de Histograma:
        A equalização de histograma é uma técnica que melhora o contraste de uma imagem, 
        redistribuindo os níveis de intensidade. Isso é especialmente útil em imagens com 
        iluminação desigual ou baixa visibilidade.

    Parâmetros:
    img (np.ndarray): Explicado acima, na função de filtro de mediana

    Retorna:
    np.ndarray: Outro array numpy representando a imagem com o histograma equalizado.
    """
    if len(img.shape) == 3:
        """
        Para casos coloridos:
        Não queremos alterar as cores de forma distrocida, e como equalizar cada canal rgb separadamente muda o equilibrio de cores,
        o correto é converter a imagem para o espaço de cores YCrCb, onde Y representa a luminância (brilho) e Cr e Cb representam as informações de cor.
        A equalização é aplicada apenas no canal Y (luminância), enquanto os canais Cr e Cb permanecem inalterados.
        Isso preserva a cor original da imagem, enquanto melhora o contraste.
        len(img.shape) dá o npmero de dimensões do array, 2-> imagem em escala de cinza, 3-> imagem colorida
        cv2.split() -> Divide a imagem em seus canais (B, G, R)
        cv2.cvtColor() -> Converte a imagem de BGR para YCrCb
        cv2.equalizeHist() -> Aplica a equalização de histograma no canal Y (luminância)
        cv2.merge() -> Junta os canais Y, Cr e Cb de volta em uma imagem
        """
        # Caso colorido: converte BGR → YCrCb, equaliza só o Y, retorna para BGR
        y, u, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
        y_eq = cv2.equalizeHist(y)
        return cv2.cvtColor(cv2.merge([y_eq, u, v]), cv2.COLOR_YCrCb2BGR)
    else:
        # Caso cinza: equaliza diretamente
        return cv2.equalizeHist(img)

