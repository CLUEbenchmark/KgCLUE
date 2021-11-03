from typing import List, Optional
import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self,num_tags : int = 2, batch_first:bool = True) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        # start 到其他tag(不包含end)的得分
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        # 到其他tag(不包含start)到end的得分
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        # 从 _compute_normalizer 中 next_score = broadcast_score + self.transitions + broadcast_emissions 可以看出
        # transitions[i][j] 表示从第j个tag 到第 i 个 tag的分数
        # 更正 ：transitions[i][j] 表示从第i个tag 到第 j 个 tag的分数
        self.transitions = nn.Parameter(torch.empty(num_tags,num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        init_range = 0.1
        nn.init.uniform_(self.start_transitions,-init_range,init_range)
        nn.init.uniform_(self.end_transitions,-init_range,init_range)
        nn.init.uniform_(self.transitions, -init_range, init_range)

    def __repr__(self):
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self, emissions:torch.Tensor,
                tags:torch.Tensor = None,
                mask:Optional[torch.ByteTensor] = None,
                reduction: str = "mean") -> torch.Tensor:

        self._validate(emissions, tags = tags ,mask = mask)

        reduction = reduction.lower()
        if reduction not in ('none','sum','mean','token_mean'):
            raise ValueError(f'invalid reduction {reduction}')

        if mask is None:
            mask = torch.ones_like(tags,dtype = torch.uint8)
    # a.shape (seq_len,batch_size)
    # a[0] shape ? batch_size

        if self.batch_first:
            # emissions.shape (seq_len,batch_size,tag_num)
            emissions = emissions.transpose(0,1)
            tags = tags.transpose(0,1)
            mask = mask.transpose(0,1)

        # shape: (batch_size,)
        numerator = self._computer_score(emissions=emissions,tags=tags,mask=mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions=emissions,mask=mask)
        # shape: (batch_size,)
        llh = denominator - numerator

        if reduction == 'none':
            return llh
        elif reduction == 'sum':
            return llh.sum()
        elif reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.float().sum()

    def decode(self,emissions:torch.Tensor,
               mask : Optional[torch.ByteTensor] = None) ->List[List[int]]:
        self._validate(emissions=emissions,mask=mask)

        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2],dtype=torch.uint8)
        if self.batch_first:
            emissions = emissions.transpose(0,1)
            mask = mask.transpose(0,1)

        return self._viterbi_decode(emissions,mask)






    def _validate(self,
                  emissions:torch.Tensor,
                  tags:Optional[torch.LongTensor] = None ,
                  mask:Optional[torch.ByteTensor] = None) -> None:

        if emissions.dim() != 3:
            raise ValueError(f"emissions must have dimension of 3 , got {emissions.dim()}")
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags},'
                f'got {emissions.size(2)}'
            )

        if tags is not None:

            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of and mask must match,'
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}'
                )
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:,0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')


    def _computer_score(self,
                        emissions:torch.Tensor,
                        tags:torch.LongTensor,
                        mask:torch.ByteTensor) -> torch.Tensor:

        # batch second
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length,batch_size = tags.shape
        mask = mask.float()

        # self.start_transitions  start 到其他tag(不包含end)的得分
        score = self.start_transitions[tags[0]]

        # emissions.shape (seq_len,batch_size,tag_nums)

        score += emissions[0,torch.arange(batch_size),tags[0]]

        for i in range(1,seq_length):

            # if mask[i].sum() == 0:
            #     break

            score += self.transitions[tags[i-1], tags[i]] * mask[i]

            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]


        # 这里是为了获取每一个样本最后一个词的tag。
        # shape: (batch_size,)   每一个batch 的真实长度
        seq_ends = mask.long().sum(dim=0) - 1
        # 每个样本最火一个词的tag
        last_tags = tags[seq_ends,torch.arange(batch_size)]
        # shape: (batch_size,) 每一个样本到最后一个词的得分加上之前的score
        score += self.end_transitions[last_tags]
        return score

    def _compute_normalizer(self,
                            emissions:torch.Tensor ,
                            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)
        # shape : (batch_size,num_tag)
        # self.start_transitions  start 到其他tag(不包含end)的得分
        # start_transitions.shape tag_nums     emissions[0].shape (batch_size,tag_size)
        score = self.start_transitions + emissions[0]

        for i in range(1,seq_length):

            # shape : (batch_size,num_tag,1)
            broadcast_score = score.unsqueeze(dim=2)

            # shape: (batch_size,1,num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            next_score = broadcast_score + self.transitions + broadcast_emissions

            next_score = torch.logsumexp(next_score,dim = 1)

            score = torch.where(mask[i].unsqueeze(1),next_score,score)

        # shape (batch_size,num_tags)
        score += self.end_transitions

        # shape: (batch_size)
        return torch.logsumexp(score,dim=1)

    def _viterbi_decode(self,emissions : torch.FloatTensor ,
                        mask : torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length , batch_size = mask.shape
        # self.start_transitions  start 到其他tag(不包含end)的得分
        score = self.start_transitions + emissions[0]
        history = []


        # for i in range(1,seq_length):
        #
        #     # shape : (batch_size,num_tag,1)
        #     broadcast_score = score.unsqueeze(dim=2)
        #
        #     # shape: (batch_size,1,num_tags)
        #     broadcast_emissions = emissions[i].unsqueeze(1)
        #
        #     next_score = broadcast_score + self.transitions + broadcast_emissions
        #
        #     next_score = torch.logsumexp(next_score,dim = 1)
        #
        #     score = torch.where(mask[i].unsqueeze(1),next_score,score)


        for i in range(1,seq_length):
            broadcast_score = score.unsqueeze(2)

            broadcast_emission = emissions[i].unsqueeze(1)

            next_score = broadcast_score + self.transitions + broadcast_emission

            next_score, indices = next_score.max(dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)

            history.append(indices)

        score += self.end_transitions

        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):

            _,best_last_tag = score[idx].max(dim = 0)

            best_tags= [best_last_tag.item()]

            # history[:seq_ends[idx]].shape  (seq_ends[idx])

            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)
        return best_tags_list