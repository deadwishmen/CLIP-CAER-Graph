from torch import nn
from models.Temporal_Model import *
from models.Prompt_Learner import *
from models.Graph_Model import *
import copy

class GenerateModel(nn.Module):
    def __init__(self, input_text, clip_model, args):
        super().__init__()
        self.args = args
        self.input_text = input_text
        self.prompt_learner = PromptLearner(input_text, clip_model, args)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual
        self.temporal_net = Temporal_Transformer_Cls(num_patches=16,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)
        
        self.temporal_net_body = Temporal_Transformer_Cls(num_patches=16,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)
        
        # self.temporal_net_graph = Temporal_Transformer_Cls(num_patches=16,
        #                                              input_dim=256,
        #                                              depth=args.temporal_layers,
        #                                              heads=8,
        #                                              mlp_dim=1024,
        #                                              dim_head=64)
        
        # self.gnn_encoder = GraphContextEncoder(input_dim=512, output_dim=256)

        self.clip_model_ = clip_model
        self.project_fc = nn.Linear(512 + 512, 512)
        
    def forward(self, image_face, image_body, x, pos, edge_index, edge_attr, batch):
        ################# Visual Part #################
        # Face Part
        n, t, c, h, w = image_face.shape
        image_face = image_face.contiguous().view(-1, c, h, w)
        image_face_features = self.image_encoder(image_face.type(self.dtype))
        image_face_features = image_face_features.contiguous().view(n, t, -1)
        video_face_features = self.temporal_net(image_face_features)  # (n, 512)
        
        # Body Part
        n, t, c, h, w = image_body.shape
        image_body = image_body.contiguous().view(-1, c, h, w)
        image_body_features = self.image_encoder(image_body.type(self.dtype))
        image_body_features = image_body_features.contiguous().view(n, t, -1)
        video_body_features = self.temporal_net_body(image_body_features)  # (n, 512)

        # Graph Part
        # graph_embeddings = self.gnn_encoder(x, edge_index, edge_attr, batch)  # (n * t, 256)
        # graph_embeddings = graph_embeddings.contiguous().view(n, t, 256)
        # video_graph_features = self.temporal_net_graph(graph_embeddings)  # (n, 256)

        # Concatenate the three parts
        video_features = torch.cat((video_face_features, video_body_features), dim=-1)
        video_features = self.project_fc(video_features)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        ################# Text Part ###################
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        ###############################################

        output = video_features @ text_features.t() / 0.01
        return output