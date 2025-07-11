"""
Sign Configuration and Classification
Define las señas y sus características para la recolección
"""


class SignConfig:
    """Configuración y clasificación de señas LSP"""
    
    def __init__(self):
        # Clasificación de señas por tipo - Optimizada para GRU
        self.sign_types = {
            'static_one_hand': {
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
            },
            'dynamic_one_hand': {
                'J', 'Z', 'Ñ', 'RR', 'LL'
            },
            'static_two_hands': {
                'AMOR', 'CASA', 'FAMILIA', 'ESCUELA'
            },
            'dynamic_two_hands': {
                'HOLA', 'GRACIAS', 'POR FAVOR', 'ADIÓS', 'CÓMO ESTÁS'
            },
            'phrases': {
                'BUENOS DÍAS', 'BUENAS NOCHES', 'MUCHO GUSTO', 'DE NADA'
            }
        }
        
        # Configuración específica por tipo de seña
        self.type_config = {
            'static_one_hand': {
                'expected_hands': 1,
                'movement_threshold': 0.012,
                'min_stability': 0.8,
                'description': 'Seña estática con una mano'
            },
            'dynamic_one_hand': {
                'expected_hands': 1,
                'movement_threshold': 0.025,
                'min_stability': 0.6,
                'description': 'Seña dinámica con una mano'
            },
            'static_two_hands': {
                'expected_hands': 2,
                'movement_threshold': 0.015,
                'min_stability': 0.8,
                'coordination_required': True,
                'description': 'Seña estática con dos manos'
            },
            'dynamic_two_hands': {
                'expected_hands': 2,
                'movement_threshold': 0.030,
                'min_stability': 0.6,
                'coordination_required': True,
                'description': 'Seña dinámica con dos manos'
            },
            'phrases': {
                'expected_hands': 2,
                'movement_threshold': 0.035,
                'min_stability': 0.5,
                'coordination_required': True,
                'description': 'Frase completa con múltiples gestos'
            }
        }
        
        # Instrucciones específicas para cada seña
        self.sign_instructions = {
            # Letras estáticas
            'A': 'Puño cerrado con pulgar al costado',
            'B': 'Mano extendida, dedos juntos, pulgar doblado',
            'C': 'Forma de C con la mano',
            'D': 'Índice extendido, otros dedos doblados tocando pulgar',
            'E': 'Dedos doblados tocando pulgar',
            'F': 'Índice y pulgar en círculo, otros dedos extendidos',
            'G': 'Índice y pulgar extendidos horizontalmente',
            'H': 'Índice y medio extendidos horizontalmente',
            'I': 'Meñique extendido, otros dedos doblados',
            'K': 'Índice y medio extendidos en V, pulgar entre ellos',
            'L': 'Índice y pulgar en L',
            'M': 'Pulgar entre medio y anular',
            'N': 'Pulgar entre índice y medio',
            'O': 'Dedos en forma de O',
            'P': 'Como K pero apuntando hacia abajo',
            'Q': 'Índice y pulgar hacia abajo',
            'R': 'Índice y medio cruzados',
            'S': 'Puño cerrado con pulgar delante',
            'T': 'Pulgar entre índice y medio (puño)',
            'U': 'Índice y medio extendidos juntos',
            'V': 'Índice y medio en V',
            'W': 'Índice, medio y anular extendidos',
            'X': 'Índice en forma de gancho',
            'Y': 'Pulgar y meñique extendidos',
            
            # Letras dinámicas
            'J': 'Forma de J dibujando la letra en el aire',
            'Z': 'Forma de Z dibujando la letra en el aire',
            'Ñ': 'N con movimiento de virgulilla',
            'RR': 'R con vibración',
            'LL': 'L con movimiento lateral',
            
            # Palabras básicas
            'AMOR': 'Abrazo cruzando brazos sobre el pecho',
            'CASA': 'Formar techo con ambas manos',
            'FAMILIA': 'F con ambas manos, movimiento circular',
            'ESCUELA': 'E con movimiento de escribir',
            
            # Saludos
            'HOLA': 'Mano abierta con movimiento de saludo',
            'GRACIAS': 'Mano en barbilla, mover hacia adelante',
            'POR FAVOR': 'Mano en pecho, movimiento circular',
            'ADIÓS': 'Mano abierta con movimiento de despedida',
            'CÓMO ESTÁS': 'Secuencia de gestos interrogativos',
            
            # Frases
            'BUENOS DÍAS': 'Gesto de sol y saludo matutino',
            'BUENAS NOCHES': 'Gesto de luna y despedida nocturna',
            'MUCHO GUSTO': 'Apretón de manos simbólico',
            'DE NADA': 'Gesto de cortesía con manos abiertas'
        }
    
    def get_all_signs(self):
        """Obtiene lista de todas las señas disponibles"""
        all_signs = set()
        for signs in self.sign_types.values():
            all_signs.update(signs)
        return sorted(list(all_signs))
    
    def classify_sign_type(self, sign):
        """Clasifica una seña según su tipo"""
        for category, signs in self.sign_types.items():
            if sign in signs:
                return category
        return 'unknown'
    
    def get_sign_config(self, sign):
        """Obtiene configuración específica para una seña"""
        sign_type = self.classify_sign_type(sign)
        config = self.type_config.get(sign_type, {})
        
        # Agregar información específica de la seña
        config['instructions'] = self.sign_instructions.get(sign, 'Sin instrucciones específicas')
        config['sign_type'] = sign_type
        
        return config
    
    def get_signs_by_category(self):
        """Obtiene señas agrupadas por categoría con nombres descriptivos"""
        categories = {
            'Letras estáticas (1 mano)': sorted(list(self.sign_types['static_one_hand'])),
            'Letras dinámicas (1 mano)': sorted(list(self.sign_types['dynamic_one_hand'])),
            'Palabras básicas (2 manos)': sorted(list(self.sign_types['static_two_hands'])),
            'Saludos y cortesía (2 manos)': sorted(list(self.sign_types['dynamic_two_hands'])),
            'Frases completas': sorted(list(self.sign_types['phrases']))
        }
        return categories
    
    def get_recommended_sequence_count(self, sign_type):
        """Obtiene el número recomendado de secuencias por tipo"""
        recommendations = {
            'static_one_hand': 30,      # Menos variabilidad
            'dynamic_one_hand': 40,     # Más variabilidad en movimiento
            'static_two_hands': 35,     # Coordinación entre manos
            'dynamic_two_hands': 50,    # Alta variabilidad
            'phrases': 60,              # Máxima complejidad
            'unknown': 30
        }
        return recommendations.get(sign_type, 30)
    
    def validate_sign_execution(self, hands_info, sign_config):
        """Valida si la ejecución de la seña es correcta"""
        issues = []
        
        expected_hands = sign_config.get('expected_hands', 1)
        detected_hands = hands_info.get('count', 0)
        
        if detected_hands < expected_hands:
            issues.append(f"Se esperaban {expected_hands} manos, detectadas {detected_hands}")
        elif detected_hands > expected_hands and expected_hands == 1:
            issues.append("Detectadas más manos de las necesarias")
        
        # Verificar coordinación si es requerida
        if sign_config.get('coordination_required', False) and detected_hands == 2:
            # Esta validación se haría en el motion_analyzer
            pass
        
        return issues
    
    def get_learning_tips(self, sign):
        """Obtiene consejos de aprendizaje para una seña específica"""
        sign_type = self.classify_sign_type(sign)
        
        general_tips = {
            'static_one_hand': [
                "Mantén la mano estable y la forma clara",
                "Asegúrate de que todos los dedos estén en la posición correcta",
                "Evita movimientos innecesarios"
            ],
            'dynamic_one_hand': [
                "Haz el movimiento de forma fluida y controlada",
                "Mantén la forma de la mano consistente durante el movimiento",
                "El movimiento debe ser claro y visible"
            ],
            'static_two_hands': [
                "Coordina ambas manos simétricamente",
                "Mantén ambas manos estables",
                "Asegúrate de que ambas manos sean visibles"
            ],
            'dynamic_two_hands': [
                "Coordina el movimiento de ambas manos",
                "Mantén el ritmo constante",
                "El movimiento debe ser simétrico cuando corresponda"
            ],
            'phrases': [
                "Divide la frase en partes más pequeñas",
                "Practica las transiciones entre gestos",
                "Mantén un ritmo natural de comunicación"
            ]
        }
        
        tips = general_tips.get(sign_type, ["Practica regularmente", "Mantén movimientos claros"])
        
        # Agregar tip específico si existe
        if sign in self.sign_instructions:
            tips.insert(0, f"Instrucción específica: {self.sign_instructions[sign]}")
        
        return tips
