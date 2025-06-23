def reorganize_episode_to_CoA(episode):
    # episode is a list, each element is a dict. keys are 'pixel_values', 'input_ids', 'labels', 'dataset_name'. length of input_ids and labels are the same, which is len(text) + 7 + eos token.
    initial_img = episode[0]['pixel_values']
    initial_text = episode[0]['input_ids'][:-8]
    dataset_name = episode[0]['dataset_name']
    action_chain = []
    action_sperate_token =  DEFAULT_ACT_TOKEN
    action_sperate_token_id = 32001

    for i in range(len(episode)):
        text = episode[i]['input_ids'][:-8]
        assert torch.equal(text, initial_text), f"Text at position {i} differs from first text"
        action = episode[i]['input_ids'][-8:-1] # a tensor of shape (7,), like tensor([31867, 31884, 31872, 31891, 31902, 31928, 31744])
        action_chain.extend(action.tolist())  # Add the action tokens
        action_chain.append(action_sperate_token_id)  # Add separator token after each action
    
    # Convert to tensor
    action_chain = torch.tensor(action_chain)
    
    # Return initial image, text and the full action chain
    return {
        'pixel_values': initial_img,
        'input_ids': torch.cat([initial_text, action_chain]),
        'labels': torch.cat([initial_text, action_chain]),
        'dataset_name': dataset_name
    }